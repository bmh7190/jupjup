"""LOST112 XML 응답을 도메인 레코드로 정규화한다."""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, datetime

from ...domain.models import RecordSource, SearchRecord
from .definitions import ApiDefinition, NO_IMAGE_MARKERS


def text(parent: ET.Element, *tags: str) -> str:
    for tag in tags:
        value = parent.findtext(f".//{tag}")
        if value:
            return value.strip()
    return ""


def parse_date(value: str) -> date | None:
    cleaned = value.strip().replace(".", "-").replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            pass
    return None


def to_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_detail_url(
    atc_id: str,
    sequence: str | None,
    source: RecordSource,
) -> str | None:
    if not atc_id or source == RecordSource.POLICE_LOST:
        return None
    params = {
        "cvlcptId": "MW-201",
        "pkupCmdtyMngId": atc_id,
    }
    if sequence:
        params["sortSn"] = sequence
    return (
        "https://minwon24.police.go.kr/cvlcpt/selectFindListDetail.do?"
        f"{urllib.parse.urlencode(params)}"
    )


def parse_record(item: ET.Element, definition: ApiDefinition) -> SearchRecord:
    is_lost = definition.record_type == "lost"
    atc_id = text(item, "atcId", "ATC_ID")
    sequence = text(item, "fdSn", "FD_SN") or None
    image_url = text(item, "lstFilePathImg", "fdFilePathImg") or None
    if image_url and any(marker in image_url for marker in NO_IMAGE_MARKERS):
        image_url = None

    return SearchRecord(
        source=definition.source,
        record_type=definition.record_type,
        atc_id=atc_id,
        sequence=sequence,
        item_name=text(item, "lstPrdtNm" if is_lost else "fdPrdtNm"),
        category=text(item, "prdtClNm") or None,
        event_date=parse_date(text(item, "lstYmd" if is_lost else "fdYmd")),
        event_time=text(item, "lstHor" if is_lost else "fdHor") or None,
        event_place=text(item, "lstPlace" if is_lost else "fdPlace") or None,
        custody_place=text(item, "depPlace") or None,
        color=text(item, "clrNm") or None,
        subject=text(item, "lstSbjt" if is_lost else "fdSbjt") or None,
        description=text(item, "uniq") or None,
        image_url=image_url,
        organization_name=text(item, "orgNm") or None,
        telephone=text(item, "tel") or None,
        status=text(item, "csteSteNm") or None,
        detail_url=build_detail_url(atc_id, sequence, definition.source),
    )
