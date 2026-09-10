"""경찰청 분실물·습득물 Open API 3종 클라이언트."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping

from .models import ApiSearchResponse, LostItemQuery, RecordSource, SearchRecord


BASE_URL = "https://apis.data.go.kr/1320000"
NO_IMAGE_MARKERS = ("img02_no_img.gif", "img04_no_img.gif")


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
        list_operation="getLostGoodsInfoAccTpNmCstdyPlace",
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
    ) -> None:
        self.service_key = service_key
        self.timeout_seconds = timeout_seconds
        self.page_size = page_size
        self.detail_limit = detail_limit

    def search_all(
        self, query: LostItemQuery
    ) -> tuple[list[ApiSearchResponse], dict[str, str]]:
        """API 세 개를 독립적으로 호출하고 일부 실패도 결과와 함께 돌려준다."""
        if not query.item_name:
            raise ValueError("API 검색에는 item_name이 필요합니다.")

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

    def search(self, definition: ApiDefinition, query: LostItemQuery) -> ApiSearchResponse:
        params = {
            "serviceKey": self.service_key,
            "pageNo": "1",
            "numOfRows": str(self.page_size),
            definition.item_param: query.item_name or "",
        }
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
        encoded = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{url}?{encoded}", headers={"User-Agent": "lost112-agent/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
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

