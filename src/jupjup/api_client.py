"""경찰청 분실물·습득물 Open API 3종 클라이언트."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import calendar
import time
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Mapping

from .models import ApiSearchResponse, LostItemQuery, RecordSource, SearchRecord, SearchScope


BASE_URL = "https://apis.data.go.kr/1320000"
NO_IMAGE_MARKERS = ("img02_no_img.gif", "img04_no_img.gif")
REQUEST_DEADLINE: ContextVar[float | None] = ContextVar("request_deadline", default=None)
PERIOD_OPERATIONS = {
    RecordSource.POLICE_LOST: "getLostGoodsInfoAccToClAreaPd",
    RecordSource.POLICE_FOUND: "getLosfundInfoAccToClAreaPd",
    RecordSource.PORTAL_FOUND: "getPtLosfundInfoAccToClAreaPd",
}


def build_search_windows(lost_date: date, today: date | None = None) -> list[tuple[date, date]]:
    """분실일 포함 7일, 다음 7일, 이어지는 한 달. 끝 날짜는 포함한다."""
    today = today or datetime.now(ZoneInfo("Asia/Seoul")).date()
    if lost_date > today:
        raise ValueError("분실일은 오늘 이후일 수 없습니다.")
    third_start = lost_date + timedelta(days=14)
    year = third_start.year + (third_start.month == 12)
    month = third_start.month % 12 + 1
    next_month = date(year, month, min(third_start.day, calendar.monthrange(year, month)[1]))
    windows = [(lost_date, lost_date + timedelta(days=6)),
               (lost_date + timedelta(days=7), third_start - timedelta(days=1)),
               (third_start, next_month - timedelta(days=1))]
    return [(start, min(end, today)) for start, end in windows if start <= today]


@dataclass(frozen=True)
class ApiDefinition:
    source: RecordSource
    service: str
    list_operation: str
    detail_operation: str
    item_param: str
    place_param: str
    record_type: str

    @property
    def list_url(self) -> str:
        return f"{BASE_URL}/{self.service}/{self.list_operation}"

    @property
    def detail_url(self) -> str:
        return f"{BASE_URL}/{self.service}/{self.detail_operation}"


API_DEFINITIONS = (
    ApiDefinition(
        source=RecordSource.POLICE_LOST,
        service="LostGoodsInfoInqireService",
        # 명칭/보관장소 조회는 현재 기관 API에서 resultCode=04를 반환한다.
        # 분류/지역/기간 조회는 동일한 승인 키로 정상 응답한다.
        list_operation="getLostGoodsInfoAccToClAreaPd",
        detail_operation="getLostGoodsDetailInfo",
        item_param="LST_PRDT_NM",
        place_param="LST_PLACE",
        record_type="lost",
    ),
    ApiDefinition(
        source=RecordSource.POLICE_FOUND,
        service="LosfundInfoInqireService",
        list_operation="getLosfundInfoAccTpNmCstdyPlace",
        detail_operation="getLosfundDetailInfo",
        item_param="PRDT_NM",
        place_param="DEP_PLACE",
        record_type="found",
    ),
    ApiDefinition(
        source=RecordSource.PORTAL_FOUND,
        service="LosPtfundInfoInqireService",
        list_operation="getPtLosfundInfoAccTpNmCstdyPlace",
        detail_operation="getPtLosfundDetailInfo",
        item_param="PRDT_NM",
        place_param="DEP_PLACE",
        record_type="found",
    ),
)


# 경찰청 공통코드의 상위 물품 분류. 자연어가 더 구체적일 수 있으므로
# 각 분류의 대표 별칭도 함께 둔다. 코드 조회 API를 승인받으면 이 표를
# 시작 시 동적으로 갱신하는 방식으로 확장할 수 있다.
PRODUCT_CATEGORY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PRH000", ("카드지갑", "반지갑", "장지갑", "지갑")),
    ("PRJ000", ("스마트폰", "핸드폰", "휴대폰", "아이폰", "갤럭시")),
    ("PRI000", ("노트북", "랩톱", "컴퓨터")),
    ("PRA000", ("백팩", "배낭", "가방")),
    ("PRP000", ("신용카드", "체크카드", "교통카드", "카드")),
    (
        "PRN000",
        ("주민등록증", "운전면허증", "면허증", "여권", "신분증", "증명서"),
    ),
    ("PRG000", ("이어폰", "카메라", "전자기기", "전자제품")),
    ("PRK000", ("외투", "재킷", "자켓", "옷", "의류")),
    ("PRO000", ("반지", "목걸이", "귀걸이", "시계", "귀금속")),
    ("PRB000", ("책", "도서")),
    ("PRC000", ("서류", "문서")),
    ("PRL000", ("현금", "수표", "외화")),
    ("PRF000", ("자동차열쇠", "차키", "자동차번호판", "내비게이션")),
    ("PRE000", ("스포츠용품", "운동용품")),
    ("PRR000", ("악기",)),
    ("PRM000", ("상품권", "어음", "채권", "유가증권")),
    ("PRQ000", ("쇼핑백",)),
    ("PRD000", ("산업용품",)),
)


def _product_category_codes(text: str) -> tuple[str | None, str | None]:
    """일반 물품명을 경찰청 상·하위 분류코드로 변환한다."""
    normalized = text.replace(" ", "").lower()
    for upper_code, aliases in PRODUCT_CATEGORY_ALIASES:
        if any(alias.lower() in normalized for alias in aliases):
            if upper_code == "PRH000":
                if "여성" in normalized:
                    return upper_code, "PRH100"
                if "남성" in normalized:
                    return upper_code, "PRH200"
            return upper_code, None
    return None, None


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
        self, query: LostItemQuery
    ) -> tuple[list[ApiSearchResponse], dict[str, str]]:
        """API 세 개를 독립적으로 호출하고 일부 실패도 결과와 함께 돌려준다."""
        if not query.item_name:
            raise ValueError("API 검색에는 item_name이 필요합니다.")
        if query.lost_date:
            return self._search_dated(query)

        responses: list[ApiSearchResponse] = []
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(API_DEFINITIONS)) as executor:
            futures = {
                executor.submit(self.search, definition, query): definition
                for definition in API_DEFINITIONS
            }
            for future in as_completed(futures):
                definition = futures[future]
                try:
                    responses.append(future.result())
                except Exception as exc:
                    errors[definition.source.label] = str(exc)

        order = {definition.source: index for index, definition in enumerate(API_DEFINITIONS)}
        responses.sort(key=lambda response: order[response.source])
        return responses, errors

    def _search_dated(self, query: LostItemQuery) -> tuple[list[ApiSearchResponse], dict[str, str]]:
        assert query.lost_date is not None
        windows = build_search_windows(query.lost_date)
        deadline = time.monotonic() + self.search_budget_seconds
        combined = {d.source: ApiSearchResponse(source=d.source, total_count=0, records=[]) for d in API_DEFINITIONS}
        errors: dict[str, str] = {}
        seen: set[tuple[RecordSource, str, str | None]] = set()
        for start, end in windows:
            with ThreadPoolExecutor(max_workers=len(API_DEFINITIONS)) as executor:
                futures = [executor.submit(self._search_window, d, query, start, end, deadline) for d in API_DEFINITIONS]
                for future in as_completed(futures):
                    result = future.result()
                    target = combined[result.source]
                    target.total_count += result.total_count
                    target.search_scopes.extend(result.search_scopes)
                    for scope in result.search_scopes:
                        if scope.error:
                            previous = errors.get(result.source.label, "")
                            errors[result.source.label] = (previous + f" {start}~{end}: {scope.error}").strip()
                    for record in result.records:
                        key = (record.source, record.atc_id, record.sequence)
                        if key not in seen:
                            seen.add(key)
                            target.records.append(record)
            if sum(len(r.records) for r in combined.values() if r.source != RecordSource.POLICE_LOST) >= 5:
                break
            if time.monotonic() >= deadline:
                break
        return list(combined.values()), errors

    def _search_window(self, definition: ApiDefinition, query: LostItemQuery,
                       start: date, end: date, deadline: float) -> ApiSearchResponse:
        scope = SearchScope(source=definition.source, start_date=start, end_date=end)
        result = ApiSearchResponse(source=definition.source, total_count=0, records=[], search_scopes=[scope])
        url = f"{BASE_URL}/{definition.service}/{PERIOD_OPERATIONS[definition.source]}"
        seen: set[tuple[str, str | None]] = set()
        token = REQUEST_DEADLINE.set(deadline)
        try:
            for page in range(1, self.max_pages_per_window + 1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("전체 조회 시간 제한")
                root = self._request_xml(url, {"serviceKey": self.service_key,
                    "START_YMD": start.strftime("%Y%m%d"), "END_YMD": end.strftime("%Y%m%d"),
                    "pageNo": str(page), "numOfRows": str(self.page_size)})
                self._ensure_success(root, definition.source)
                result.total_count = scope.total_count = _to_int(_text(root, "totalCount"))
                records = [self._parse_record(item, definition) for item in root.findall(".//item")]
                scope.pages_completed = page
                scope.retrieved_count += len(records)
                for record in records:
                    if record.event_date is None or not start <= record.event_date <= end:
                        scope.error = "요청 기간 밖 또는 날짜 미상 자료 제외"
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
                scope.error = "페이지 제한으로 일부만 조회"
            # 기간/물품 필터를 통과한 후보 중 점수가 높은 항목부터 상세를 보완한다.
            from .matcher import LostItemMatcher
            if definition.record_type == "lost":
                detail_records = result.records[:self.detail_limit]
            else:
                ranked = LostItemMatcher().rank(query, result.records, limit=self.detail_limit)
                detail_records = [candidate.record for candidate in ranked]
            detail_keys = {(record.atc_id, record.sequence) for record in detail_records}
            for index, record in enumerate(result.records):
                if (record.atc_id, record.sequence) in detail_keys:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("상세 조회 시간 제한")
                    detail = self._fetch_and_merge_detail(definition, record)
                    if detail.event_date is not None and start <= detail.event_date <= end:
                        result.records[index] = detail
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
        except (Lost112ApiError, urllib.error.URLError, ET.ParseError):
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

    @staticmethod
    def _parse_record(item: ET.Element, definition: ApiDefinition) -> SearchRecord:
        is_lost = definition.record_type == "lost"
        atc_id = _text(item, "atcId", "ATC_ID")
        sequence = _text(item, "fdSn", "FD_SN") or None
        image_url = _text(item, "lstFilePathImg", "fdFilePathImg") or None
        if image_url and any(marker in image_url for marker in NO_IMAGE_MARKERS):
            image_url = None

        return SearchRecord(
            source=definition.source,
            record_type=definition.record_type,
            atc_id=atc_id,
            sequence=sequence,
            item_name=_text(item, "lstPrdtNm" if is_lost else "fdPrdtNm"),
            category=_text(item, "prdtClNm") or None,
            event_date=_parse_date(_text(item, "lstYmd" if is_lost else "fdYmd")),
            event_time=_text(item, "lstHor" if is_lost else "fdHor") or None,
            event_place=_text(item, "lstPlace" if is_lost else "fdPlace") or None,
            custody_place=_text(item, "depPlace") or None,
            color=_text(item, "clrNm") or None,
            subject=_text(item, "lstSbjt" if is_lost else "fdSbjt") or None,
            description=_text(item, "uniq") or None,
            image_url=image_url,
            organization_name=_text(item, "orgNm") or None,
            telephone=_text(item, "tel") or None,
            status=_text(item, "csteSteNm") or None,
            detail_url=_build_detail_url(atc_id, sequence),
        )


def _text(parent: ET.Element, *tags: str) -> str:
    for tag in tags:
        value = parent.findtext(f".//{tag}")
        if value:
            return value.strip()
    return ""


def _parse_date(value: str) -> date | None:
    cleaned = value.strip().replace(".", "-").replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            pass
    return None


def _to_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_detail_url(atc_id: str, sequence: str | None) -> str | None:
    if not atc_id:
        return None
    params = {"ATC_ID": atc_id}
    if sequence:
        params["FD_SN"] = sequence
    return f"https://minwon24.police.go.kr/main.do?{urllib.parse.urlencode(params)}"
