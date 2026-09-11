"""경찰청 분실물·습득물 Open API 3종 HTTP 클라이언트."""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from datetime import date, timedelta
from typing import Collection, Mapping

from ...domain.models import ApiSearchResponse, LostItemQuery, RecordSource, SearchRecord, SearchScope
from .definitions import (
    API_DEFINITIONS,
    BASE_URL,
    FOUND_RECORD_SOURCES,
    PERIOD_OPERATIONS,
    ApiDefinition,
    _product_category_codes,
    build_search_windows,
)
from .parser import parse_record, text as _text, to_int as _to_int

REQUEST_DEADLINE: ContextVar[float | None] = ContextVar("request_deadline", default=None)
class Lost112ApiError(RuntimeError):
    pass


class Lost112ApiClient:
    def __init__(
        self,
        service_key: str,
        *,
        timeout_seconds: float = 15,
        page_size: int = 10,
        detail_limit: int = 5,
        max_pages_per_window: int = 10,
        search_budget_seconds: float = 30,
    ) -> None:
        self.service_key = service_key
        self.timeout_seconds = timeout_seconds
        self.page_size = page_size
        self.detail_limit = detail_limit
        self.max_pages_per_window = max(1, max_pages_per_window)
        self.search_budget_seconds = max(0.01, search_budget_seconds)

    def search_all(
        self,
        query: LostItemQuery,
        *,
        sources: Collection[RecordSource] | None = None,
    ) -> tuple[list[ApiSearchResponse], dict[str, str]]:
        """선택한 출처를 독립적으로 호출하고 일부 실패도 결과와 함께 돌려준다."""
        if not query.item_name:
            raise ValueError("API 검색에는 item_name이 필요합니다.")
        definitions = tuple(
            definition
            for definition in API_DEFINITIONS
            if sources is None or definition.source in sources
        )
        if not definitions:
            return [], {}
        if query.lost_date:
            return self._search_dated(query, definitions)

        responses: list[ApiSearchResponse] = []
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(definitions)) as executor:
            futures = {
                executor.submit(self.search, definition, query): definition
                for definition in definitions
            }
            for future in as_completed(futures):
                definition = futures[future]
                try:
                    responses.append(future.result())
                except Exception as exc:
                    errors[definition.source.label] = str(exc)

        order = {
            definition.source: index
            for index, definition in enumerate(definitions)
        }
        responses.sort(key=lambda response: order[response.source])
        return responses, errors

    def _search_dated(
        self,
        query: LostItemQuery,
        definitions: tuple[ApiDefinition, ...],
    ) -> tuple[list[ApiSearchResponse], dict[str, str]]:
        assert query.lost_date is not None
        windows = build_search_windows(query.lost_date)
        deadline = time.monotonic() + self.search_budget_seconds
        combined = {
            definition.source: ApiSearchResponse(
                source=definition.source,
                total_count=0,
                records=[],
            )
            for definition in definitions
        }
        seen: set[tuple[RecordSource, str, str | None]] = set()
        for start, end in windows:
            with ThreadPoolExecutor(max_workers=len(definitions)) as executor:
                futures = [
                    executor.submit(
                        self._search_window,
                        definition,
                        query,
                        start,
                        end,
                        deadline,
                    )
                    for definition in definitions
                ]
                for future in as_completed(futures):
                    result = future.result()
                    target = combined[result.source]
                    target.total_count += result.total_count
                    target.search_scopes.extend(result.search_scopes)
                    for record in result.records:
                        key = (record.source, record.atc_id, record.sequence)
                        if key not in seen:
                            seen.add(key)
                            target.records.append(record)
            found_responses = [
                response
                for response in combined.values()
                if response.source in FOUND_RECORD_SOURCES
            ]
            target_responses = found_responses or list(combined.values())
            if sum(len(response.records) for response in target_responses) >= 5:
                break
            if time.monotonic() >= deadline:
                break
        for definition in definitions:
            self._enrich_dated_details(
                definition, combined[definition.source], query, deadline
            )
        responses = list(combined.values())

        # 한 출처라도 자료를 확보했다면 전체 시간 제한은 실패가 아니라 부분
        # 성공이다. 실제 API 오류는 그대로 남겨 사용자에게 구분해 보여준다.
        if any(response.records for response in responses):
            timeout_reason = "전체 조회 시간 제한으로 목록 일부만 확인"
            for response in responses:
                for scope in response.search_scopes:
                    if scope.error and "TimeoutError" in scope.error:
                        if scope.partial_reason:
                            scope.partial_reason = (
                                f"{scope.partial_reason}; {timeout_reason}"
                            )
                        else:
                            scope.partial_reason = timeout_reason
                        scope.error = None

        errors: dict[str, str] = {}
        for response in responses:
            for scope in response.search_scopes:
                if not scope.error:
                    continue
                previous = errors.get(response.source.label, "")
                message = f"{scope.start_date}~{scope.end_date}: {scope.error}"
                errors[response.source.label] = f"{previous} {message}".strip()
        return responses, errors

    def _enrich_dated_details(
        self,
        definition: ApiDefinition,
        result: ApiSearchResponse,
        query: LostItemQuery,
        deadline: float,
    ) -> None:
        """기간 목록을 모두 모은 뒤 출처별 상위 후보만 상세 조회한다."""
        if definition.record_type == "lost":
            detail_records = result.records[:self.detail_limit]
        else:
            from ...domain.matcher import LostItemMatcher

            ranked = LostItemMatcher().rank(
                query, result.records, limit=self.detail_limit
            )
            detail_records = [candidate.record for candidate in ranked]

        detail_keys = {(record.atc_id, record.sequence) for record in detail_records}
        token = REQUEST_DEADLINE.set(deadline)
        try:
            for index, record in enumerate(result.records):
                if (record.atc_id, record.sequence) not in detail_keys:
                    continue
                if time.monotonic() >= deadline:
                    self._mark_detail_enrichment_skipped(result)
                    break
                try:
                    result.records[index] = self._fetch_and_merge_detail(
                        definition, record
                    )
                except TimeoutError:
                    self._mark_detail_enrichment_skipped(result)
                    break
        finally:
            REQUEST_DEADLINE.reset(token)

    @staticmethod
    def _mark_detail_enrichment_skipped(result: ApiSearchResponse) -> None:
        if not result.search_scopes:
            return
        scope = result.search_scopes[-1]
        reason = "전체 시간 예산 소진으로 상세 조회 일부 생략"
        if scope.partial_reason:
            scope.partial_reason = f"{scope.partial_reason}; {reason}"
        else:
            scope.partial_reason = reason

    def _search_window(self, definition: ApiDefinition, query: LostItemQuery,
                       start: date, end: date, deadline: float) -> ApiSearchResponse:
        scope = SearchScope(source=definition.source, start_date=start, end_date=end)
        result = ApiSearchResponse(source=definition.source, total_count=0, records=[], search_scopes=[scope])
        url = f"{BASE_URL}/{definition.service}/{PERIOD_OPERATIONS[definition.source]}"
        seen: set[tuple[str, str | None]] = set()
        token = REQUEST_DEADLINE.set(deadline)
        category_text = " ".join(value for value in (query.category, query.item_name) if value)
        upper_code, lower_code = _product_category_codes(category_text)

        try:
            for page in range(1, self.max_pages_per_window + 1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("전체 조회 시간 제한")

                params = {
                    "serviceKey": self.service_key,
                    "START_YMD": start.strftime("%Y%m%d"),
                    "END_YMD": end.strftime("%Y%m%d"),
                    "pageNo": str(page),
                    "numOfRows": str(self.page_size)
                }
                if upper_code:
                    params["PRDT_CL_CD_01"] = upper_code
                if lower_code:
                    params["PRDT_CL_CD_02"] = lower_code

                root = self._request_xml(url, params)
                self._ensure_success(root, definition.source)
                result.total_count = scope.total_count = _to_int(_text(root, "totalCount"))
                records = [self._parse_record(item, definition) for item in root.findall(".//item")]
                scope.pages_completed = page
                scope.retrieved_count += len(records)
                for record in records:
                    if record.event_date is None or not start <= record.event_date <= end:
                        scope.partial_reason = "요청 기간 밖 또는 날짜 미상 자료 제외"
                        continue
                    # 기간 API는 PRDT_NM을 무시하므로 명칭/분류는 로컬에서 검사한다.
                    needle = (query.item_name or "").replace(" ", "").casefold()
                    haystack = f"{record.item_name} {record.category or ''}".replace(" ", "").casefold()
                    key = (record.atc_id, record.sequence)
                    if record.atc_id and needle in haystack and key not in seen:
                        seen.add(key)
                        result.records.append(record)
                if scope.retrieved_count >= result.total_count:
                    scope.complete = scope.error is None
                    break
                if not records:
                    scope.error = "전체 건수에 도달하기 전에 빈 페이지 반환"
                    break
            else:
                scope.partial_reason = "설정된 페이지 상한에 도달"
        except Exception as exc:
            # 키를 포함할 수 있는 원본 예외/URL을 결과에 저장하지 않는다.
            scope.complete = False
            scope.error = f"조회 중단 ({type(exc).__name__})"
        finally:
            REQUEST_DEADLINE.reset(token)
        return result

    def search(self, definition: ApiDefinition, query: LostItemQuery) -> ApiSearchResponse:
        params = self._build_list_params(definition, query)
        root = self._request_xml(definition.list_url, params)
        self._ensure_success(root, definition.source)
        total_count = _to_int(_text(root, "totalCount"))
        records = [
            self._parse_record(item, definition)
            for item in root.findall(".//item")
        ]

        # 목록에는 습득장소·특이사항이 없으므로 상위 일부만 상세 조회한다.
        for index, record in enumerate(records[: self.detail_limit]):
            if not record.atc_id:
                continue
            records[index] = self._fetch_and_merge_detail(definition, record)

        return ApiSearchResponse(source=definition.source, total_count=total_count, records=records)

    def _build_list_params(
        self, definition: ApiDefinition, query: LostItemQuery
    ) -> dict[str, str]:
        params = {
            "serviceKey": self.service_key,
            "pageNo": "1",
            "numOfRows": str(self.page_size),
        }
        if definition.source != RecordSource.POLICE_LOST:
            params[definition.item_param] = query.item_name or ""
            return params

        # 이 오퍼레이션은 START_YMD가 없으면 HTTP 200 안에 resultCode=04를 반환한다.
        today = date.today()
        start_date = (
            query.lost_date
            if query.lost_date and query.lost_date <= today
            else today - timedelta(days=90)
        )
        params["START_YMD"] = start_date.strftime("%Y%m%d")
        params["END_YMD"] = today.strftime("%Y%m%d")

        category_text = " ".join(
            value for value in (query.category, query.item_name) if value
        )
        upper_code, lower_code = _product_category_codes(category_text)
        if upper_code:
            params["PRDT_CL_CD_01"] = upper_code
        if lower_code:
            params["PRDT_CL_CD_02"] = lower_code
        return params

    def _fetch_and_merge_detail(
        self, definition: ApiDefinition, record: SearchRecord
    ) -> SearchRecord:
        params = {"serviceKey": self.service_key, "ATC_ID": record.atc_id}
        if record.sequence:
            params["FD_SN"] = record.sequence
        try:
            root = self._request_xml(definition.detail_url, params)
            self._ensure_success(root, definition.source)
            item = root.find(".//item")
            if item is None:
                return record
            detail = self._parse_record(item, definition)
        except (Lost112ApiError, urllib.error.URLError, TimeoutError, ET.ParseError):
            return record

        updates = {
            field: getattr(detail, field)
            for field in SearchRecord.model_fields
            if getattr(detail, field) not in (None, "")
        }
        return record.model_copy(update=updates)

    def _request_xml(self, url: str, params: Mapping[str, str]) -> ET.Element:
        deadline = REQUEST_DEADLINE.get()
        timeout = self.timeout_seconds
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("전체 조회 시간 제한")
        encoded = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{url}?{encoded}", headers={"User-Agent": "jupjup/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise Lost112ApiError(f"HTTP {exc.code}: {body}") from exc
        try:
            return ET.fromstring(body)
        except ET.ParseError as exc:
            preview = body.decode("utf-8", errors="replace")[:300]
            raise Lost112ApiError(f"XML 응답을 해석하지 못했습니다: {preview}") from exc

    @staticmethod
    def _ensure_success(root: ET.Element, source: RecordSource) -> None:
        code = _text(root, "resultCode", "returnReasonCode")
        message = _text(root, "resultMsg", "resultMag", "returnAuthMsg")
        if code not in {"00", "0", "NORMAL_CODE"}:
            raise Lost112ApiError(f"{source.label} 오류 {code or '코드 없음'}: {message}")


    _parse_record = staticmethod(parse_record)
