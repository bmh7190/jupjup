"""경찰민원24 제출 전에 검토할 분실신고 초안을 만든다."""

from __future__ import annotations

from datetime import date

from .models import LostReportDraft
from .privacy import mask_pii


POLICE_REPORT_GUIDE_URL = (
    "https://minwon24.police.go.kr/cvlcpt/"
    "cvlcptGdInfo.do?cvlcptId=MW-001"
)
POLICE_REPORT_URL = "https://minwon24.police.go.kr/main.do"

FIELD_LABELS = {
    "item_name": "분실 물품명",
    "lost_date": "분실 날짜",
    "lost_place": "구체적인 분실 장소",
    "lost_time": "분실 시간대",
    "region": "분실 지역",
    "color": "물품 색상",
    "size": "물품 크기",
    "brand": "브랜드 또는 제조사",
    "quantity": "분실 수량",
    "features": "식별 가능한 특징이나 내용물",
    "circumstances": "분실 전후 상황이나 이동 경로",
}


def build_lost_report_draft(
    *,
    item_name: str | None = None,
    category: str | None = None,
    lost_date: date | None = None,
    lost_time: str | None = None,
    lost_place: str | None = None,
    region: str | None = None,
    color: str | None = None,
    size: str | None = None,
    brand: str | None = None,
    quantity: int | None = None,
    features: list[str] | None = None,
    circumstances: str | None = None,
    incident_type: str = "분실",
) -> LostReportDraft:
    """입력값을 검토하고 복사 가능한 신고 문장과 보완점을 반환한다."""
    clean_item_name = _clean(item_name)
    clean_category = _clean(category)
    clean_time = _clean(lost_time)
    clean_place = _clean(lost_place)
    clean_region = _clean(region)
    clean_color = _clean(color)
    clean_size = _clean(size)
    clean_brand = _clean(brand)
    clean_features = [
        mask_pii(value.strip()) for value in features or [] if value.strip()
    ]
    clean_circumstances = mask_pii(circumstances.strip()) if circumstances else None

    values = {
        "item_name": clean_item_name,
        "lost_date": lost_date,
        "lost_place": clean_place,
        "lost_time": clean_time,
        "region": clean_region,
        "color": clean_color,
        "size": clean_size,
        "brand": clean_brand,
        "quantity": quantity,
        "features": clean_features,
        "circumstances": clean_circumstances,
    }
    essential_keys = ("item_name", "lost_date", "lost_place")
    recommended_keys = (
        "lost_time",
        "region",
        "color",
        "size",
        "brand",
        "quantity",
        "features",
        "circumstances",
    )
    missing_essential = [FIELD_LABELS[key] for key in essential_keys if not values[key]]
    missing_recommended = [FIELD_LABELS[key] for key in recommended_keys if not values[key]]

    tips = _improvement_tips(values)
    notices = [
        "이 기능은 신고서를 대신 제출하지 않으며, 최종 내용은 사용자가 직접 확인해야 합니다.",
        "경찰민원24 온라인 분실신고는 제출 후 수정할 수 없어 취소 후 다시 신고해야 합니다.",
        "경찰민원24 안내 기준으로 분실신고에 별도 구비서류나 수수료는 없습니다.",
        "카드번호·주민등록번호·연락처 전체 값은 물품 특징란에 적지 마세요.",
        "성명·생년월일·연락처 등 신고인 정보는 경찰민원24에서 본인이 직접 확인하세요.",
        "경찰민원24 메인에서 '유실물 민원 > 분실물 신고'를 선택하고 로그인해 작성하세요.",
        "SMS/이메일 수신을 허용하면 신상정보가 일치하는 습득물 입고 시 안내받을 수 있습니다.",
    ]

    normalized_incident = incident_type.replace(" ", "")
    is_theft = any(word in normalized_incident for word in ("도난", "절도", "훔침"))
    if is_theft:
        notices.insert(
            0,
            "도난은 분실물 신고 대상이 아닙니다. 가까운 경찰관서 또는 112의 안내를 받으세요.",
        )
    if clean_item_name and "자동차번호판" in clean_item_name.replace(" ", ""):
        notices.insert(0, "자동차번호판 분실은 경찰관서 방문 신고가 필요합니다.")

    ready = not missing_essential and not is_theft
    next_question = _next_question(missing_essential)
    title = f"{clean_item_name or '물품'} 분실 신고"
    copy_text = _build_copy_text(
        item_name=clean_item_name,
        category=clean_category,
        lost_date=lost_date,
        lost_time=clean_time,
        lost_place=clean_place,
        region=clean_region,
        color=clean_color,
        size=clean_size,
        brand=clean_brand,
        quantity=quantity,
        features=clean_features,
        circumstances=clean_circumstances,
    )

    return LostReportDraft(
        title=title,
        copy_text=copy_text,
        item_name=clean_item_name,
        category=clean_category,
        lost_date=lost_date,
        lost_time=clean_time,
        lost_place=clean_place,
        region=clean_region,
        color=clean_color,
        size=clean_size,
        brand=clean_brand,
        quantity=quantity,
        features=clean_features,
        circumstances=clean_circumstances,
        missing_essential_fields=missing_essential,
        missing_recommended_fields=missing_recommended,
        improvement_tips=tips,
        notices=notices,
        next_question=next_question,
        ready_for_user_review=ready,
        official_report_url=POLICE_REPORT_URL,
        official_guide_url=POLICE_REPORT_GUIDE_URL,
    )


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = mask_pii(value.strip())
    return cleaned or None


def _improvement_tips(values: dict[str, object]) -> list[str]:
    tips: list[str] = []
    if not values["lost_time"]:
        tips.append("마지막으로 물건을 확인한 시각과 분실을 알아챈 시각으로 시간 범위를 좁혀보세요.")
    if not values["lost_place"]:
        tips.append("역명·노선·차량번호·매장명·좌석처럼 다시 찾을 수 있는 장소 단서를 적어보세요.")
    elif not values["region"]:
        tips.append("분실 장소의 시·도와 시·군·구를 함께 적어 지역 검색이 가능하게 해보세요.")
    if not values["color"] or not values["size"] or not values["brand"]:
        tips.append("색상·크기·브랜드를 함께 적으면 같은 종류의 물품을 구분하기 쉽습니다.")
    if not values["quantity"]:
        tips.append("같이 잃어버린 물품이 여러 개라면 물품별 수량을 확인해보세요.")
    if not values["features"]:
        tips.append("흠집, 스티커, 재질, 내부 구성처럼 본인만 아는 특징을 추가해보세요.")
    if not values["circumstances"]:
        tips.append("마지막 확인 장소부터 분실을 알아챈 곳까지의 이동 경로를 짧게 적어보세요.")
    return tips


def _next_question(missing_essential: list[str]) -> str | None:
    if missing_essential:
        return f"신고서 초안을 만들려면 {missing_essential[0]}을(를) 알려주세요."
    return None


def _build_copy_text(
    *,
    item_name: str | None,
    category: str | None,
    lost_date: date | None,
    lost_time: str | None,
    lost_place: str | None,
    region: str | None,
    color: str | None,
    size: str | None,
    brand: str | None,
    quantity: int | None,
    features: list[str],
    circumstances: str | None,
) -> str:
    lines = [
        f"분실 물품: {item_name or '[확인 필요]'}",
        f"분실 수량: {quantity if quantity is not None else '[수량 확인 필요]'}",
        f"물품 분류: {category or item_name or '[확인 필요]'}",
        f"분실 일시: {_date_text(lost_date)} {lost_time or '[시간 확인 필요]'}",
        f"분실 장소: {' '.join(value for value in (region, lost_place) if value) or '[확인 필요]'}",
        f"색상/크기/브랜드: {' / '.join(value for value in (color, size, brand) if value) or '[확인 필요]'}",
        f"식별 특징: {', '.join(features) or '[특징 확인 필요]'}",
    ]
    if circumstances:
        lines.append(f"분실 경위: {circumstances}")
    return "\n".join(lines)


def _date_text(value: date | None) -> str:
    return value.isoformat() if value else "[날짜 확인 필요]"
