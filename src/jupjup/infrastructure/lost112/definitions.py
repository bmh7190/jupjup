"""LOST112 API 주소, 오퍼레이션과 물품 분류 정의."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from ...domain.models import RecordSource
BASE_URL = "https://apis.data.go.kr/1320000"
NO_IMAGE_MARKERS = ("img02_no_img.gif", "img04_no_img.gif")
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

FOUND_RECORD_SOURCES = frozenset(
    {RecordSource.POLICE_FOUND, RecordSource.PORTAL_FOUND}
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
