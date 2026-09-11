"""설명 가능한 규칙 기반 유사도 계산기."""

from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher
from typing import Mapping, Sequence

from .models import LostItemQuery, MatchCandidate, ScoreBreakdown, SearchRecord


DEFAULT_WEIGHTS = {
    "name": 0.30,
    "category": 0.10,
    "color": 0.15,
    "place": 0.20,
    "date": 0.15,
    "features": 0.10,
}
MIN_CANDIDATE_SCORE = 40.0

COLOR_ALIASES = {
    "검정": {"검정", "검은", "블랙", "black"},
    "흰색": {"흰", "하양", "화이트", "white"},
    "빨강": {"빨강", "붉은", "적색", "레드", "red"},
    "파랑": {"파랑", "푸른", "청색", "블루", "blue"},
    "갈색": {"갈색", "브라운", "brown"},
    "회색": {"회색", "그레이", "gray", "grey"},
    "분홍": {"분홍", "핑크", "pink"},
    "노랑": {"노랑", "노란", "옐로", "yellow"},
    "초록": {"초록", "녹색", "그린", "green"},
}

PHONE_REGION_PREFIXES = {
    "02": "서울",
    "031": "경기",
    "032": "인천",
    "033": "강원",
    "041": "충남",
    "042": "대전",
    "043": "충북",
    "044": "세종",
    "051": "부산",
    "052": "울산",
    "053": "대구",
    "054": "경북",
    "055": "경남",
    "061": "전남",
    "062": "광주",
    "063": "전북",
    "064": "제주",
}

# 이름만으로 광역 지역을 확실히 판별할 수 있는 표현만 둔다.
REGION_ALIASES = {
    "서울": {
        "서울", "강남", "강북", "강동", "서초", "송파", "마포", "용산",
        "영등포", "종로", "성북", "동대문", "광진", "관악", "동작",
        "구로", "금천", "노원", "도봉", "은평", "서대문", "양천",
    },
    "부산": {"부산", "해운대", "좌동", "명지", "서면"},
    "인천": {"인천", "간석", "부평", "송도"},
    "대구": {"대구", "수성", "달서"},
    "광주": {"광주광역시", "광주광역"},
    "대전": {"대전", "유성"},
    "울산": {"울산"},
    "세종": {"세종"},
    "경기": {"경기", "수원", "영통", "성남", "분당", "용인", "고양", "일산"},
    "강원": {"강원", "춘천", "원주", "강릉"},
    "충북": {"충북", "충청북도", "청주", "율량"},
    "충남": {"충남", "충청남도", "천안", "아산"},
    "전북": {"전북", "전북특별자치도", "전주", "익산"},
    "전남": {"전남", "전라남도", "목포", "여수", "순천"},
    "경북": {"경북", "경상북도", "포항", "경주", "구미"},
    "경남": {"경남", "경상남도", "창원", "김해", "진주"},
    "제주": {"제주"},
}

SEOUL_DISTRICT_ALIASES = {
    "강남구": {"강남구", "강남", "강남역", "역삼", "삼성", "대치", "도곡", "개포", "수서", "청담", "압구정", "신사", "논현", "세곡"},
    "서초구": {"서초구", "서초", "반포", "잠원", "방배", "양재", "내곡"},
    "송파구": {"송파구", "송파", "잠실", "문정", "가락", "석촌", "방이", "오금"},
    "성동구": {"성동구", "성동", "성수", "왕십리", "금호", "옥수"},
    "광진구": {"광진구", "광진", "건대", "구의", "자양", "화양"},
    "강동구": {"강동구", "강동", "상일", "천호", "암사", "명일", "고덕"},
}

NEIGHBORING_DISTRICTS = {
    "강남구": {"서초구", "송파구", "성동구", "광진구"},
    "서초구": {"강남구", "동작구", "관악구", "용산구"},
    "송파구": {"강남구", "강동구", "광진구"},
    "성동구": {"강남구", "광진구", "용산구", "중구", "동대문구"},
    "광진구": {"강남구", "송파구", "성동구", "강동구", "중랑구"},
    "강동구": {"송파구", "광진구"},
}

ADJACENT_REGIONS = {
    "서울": {"경기"},
    "경기": {"서울", "인천"},
    "인천": {"경기"},
    "부산": {"울산", "경남"},
    "대구": {"경북", "경남"},
    "광주": {"전남"},
    "대전": {"세종", "충남", "충북"},
    "울산": {"부산", "경남", "경북"},
    "세종": {"대전", "충남", "충북"},
    "강원": {"경기", "충북", "경북"},
    "충북": {"경기", "강원", "대전", "세종", "충남", "경북"},
    "충남": {"경기", "대전", "세종", "충북", "전북"},
    "전북": {"충남", "충북", "전남", "경남", "경북"},
    "전남": {"광주", "전북", "경남"},
    "경북": {"강원", "충북", "전북", "경남", "대구", "울산"},
    "경남": {"부산", "울산", "대구", "경북", "전북", "전남"},
    "제주": set(),
}

LOCATION_SCOPE_BY_TIER = {
    0: "direct",
    1: "district",
    2: "nearby",
    3: "region",
    4: "adjacent",
    5: "nationwide",
}

LOCATION_SCORE_CAPS = {
    0: 100.0,
    1: 95.0,
    2: 85.0,
    3: 75.0,
    4: 60.0,
    5: 45.0,
}


class LostItemMatcher:
    def rank(
        self,
        query: LostItemQuery,
        records: Sequence[SearchRecord],
        *,
        limit: int = 5,
        vision_scores: Mapping[str, float] | None = None,
    ) -> list[MatchCandidate]:
        scored = [
            (
                self.score(query, record, (vision_scores or {}).get(record.atc_id)),
                _location_tier(query, record),
            )
            for record in records
            if (
                record.record_type == "found"
                and _is_temporally_possible(query, record)
            )
        ]
        candidates = [
            (candidate, location_tier)
            for candidate, location_tier in scored
            if candidate.score >= MIN_CANDIDATE_SCORE
        ]
        if candidates and _query_region(query):
            nearest_tier = min(location_tier for _, location_tier in candidates)
            candidates = [
                (candidate, location_tier)
                for candidate, location_tier in candidates
                if location_tier == nearest_tier
            ]
        ranked = [candidate for candidate, _ in candidates]
        ranked.sort(key=lambda candidate: candidate.score, reverse=True)
        return ranked[:limit]

    def score(
        self, query: LostItemQuery, record: SearchRecord, vision_score: float | None = None
    ) -> MatchCandidate:
        component_scores = {
            "name": _text_similarity(query.item_name, record.item_name),
            "category": _text_similarity(query.category, record.category),
            "color": _color_similarity(query.color, f"{record.color or ''} {record.searchable_text}"),
            "place": max(
                _text_similarity(query.lost_place, record.event_place),
                _text_similarity(query.region, record.event_place),
                _text_similarity(query.region, record.custody_place),
                _text_similarity(query.region, record.organization_name),
                _location_similarity(query, record),
            ),
            "date": _date_similarity(query.lost_date, record.event_date),
            "features": _feature_similarity(
                [value for value in [query.brand, *query.features] if value],
                record.searchable_text,
            ),
        }

        # 값이 없는 항목의 가중치는 제외하고 남은 항목으로 100점을 다시 계산한다.
        available = {
            key: value
            for key, value in component_scores.items()
            if _query_has_value(query, key)
        }
        weights = {key: DEFAULT_WEIGHTS[key] for key in available}
        if vision_score is not None:
            available["image"] = max(0.0, min(vision_score, 1.0))
            weights["image"] = 0.15

        weight_sum = sum(weights.values()) or 1.0
        total = sum(available[key] * weights[key] for key in available) / weight_sum
        location_tier = _location_tier(query, record)
        score = round(total * 100, 1)
        if _query_region(query):
            score = min(score, LOCATION_SCORE_CAPS[location_tier])
        location_scope = (
            LOCATION_SCOPE_BY_TIER[location_tier]
            if _query_region(query)
            else "unrestricted"
        )
        breakdown = ScoreBreakdown(**component_scores, image=vision_score)
        reasons = _build_reasons(query, record, component_scores, vision_score)
        if location_scope != "unrestricted":
            reasons.append(_location_scope_reason(location_scope))
        return MatchCandidate(
            record=record,
            score=score,
            breakdown=breakdown,
            confidence=_confidence_level(
                query, score, component_scores, location_scope
            ),
            location_scope=location_scope,
            reasons=reasons,
        )


def _query_has_value(query: LostItemQuery, key: str) -> bool:
    return {
        "name": bool(query.item_name),
        "category": bool(query.category),
        "color": bool(query.color),
        "place": bool(query.lost_place or query.region),
        "date": bool(query.lost_date),
        "features": bool(query.features or query.brand),
    }[key]


def _is_temporally_possible(query: LostItemQuery, record: SearchRecord) -> bool:
    """분실 전에 습득된 물건은 동일한 물건일 수 없으므로 후보에서 제외한다."""
    if not query.lost_date or not record.event_date:
        return True
    return record.event_date >= query.lost_date


def _location_tier(query: LostItemQuery, record: SearchRecord) -> int:
    """0 직접 장소, 1 같은 구, 2 인접 구, 3 같은 시도, 4 인접 시도 순이다."""
    query_region = _query_region(query)
    if not query_region:
        return 0

    record_region = _record_region(record)
    if not record_region:
        return 5

    record_places = " ".join(
        value
        for value in (
            record.event_place,
            record.custody_place,
            record.organization_name,
        )
        if value
    )
    if _is_direct_place_match(query.lost_place, record_places):
        return 0

    query_district = _seoul_district(
        " ".join(value for value in (query.region, query.lost_place) if value),
        query_region,
    )
    record_district = _seoul_district(record_places, record_region)
    if query_district and record_district:
        if query_district == record_district:
            return 1
        if record_district in NEIGHBORING_DISTRICTS.get(query_district, set()):
            return 2

    if query_region == record_region:
        return 3
    if record_region in ADJACENT_REGIONS.get(query_region, set()):
        return 4
    return 5


def _location_similarity(query: LostItemQuery, record: SearchRecord) -> float:
    return {
        0: 1.0,
        1: 0.85,
        2: 0.65,
        3: 0.4,
        4: 0.2,
    }.get(_location_tier(query, record), 0.0)


def _is_direct_place_match(query_place: str | None, record_places: str) -> bool:
    query_text = _normalize(query_place).replace(" ", "")
    record_text = _normalize(record_places).replace(" ", "")
    if len(query_text) < 3 or len(record_text) < 3:
        return False
    return query_text in record_text or record_text in query_text


def _query_region(query: LostItemQuery) -> str | None:
    return _region_from_text(query.region or "") or _region_from_text(
        query.lost_place or ""
    )


def _record_region(record: SearchRecord) -> str | None:
    phone_region = _region_from_phone(record.telephone)
    if phone_region:
        return phone_region
    return _region_from_text(
        " ".join(
            value
            for value in (
                record.event_place,
                record.custody_place,
                record.organization_name,
            )
            if value
        )
    )


def _region_from_phone(value: str | None) -> str | None:
    digits = re.sub(r"\D", "", value or "")
    for prefix in sorted(PHONE_REGION_PREFIXES, key=len, reverse=True):
        if digits.startswith(prefix):
            return PHONE_REGION_PREFIXES[prefix]
    return None


def _region_from_text(value: str) -> str | None:
    normalized = _normalize(value).replace(" ", "")
    for region in REGION_ALIASES:
        if normalized == _normalize(region).replace(" ", ""):
            return region
    for region, aliases in REGION_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return region
    return None


def _seoul_district(value: str, region: str | None) -> str | None:
    if region != "서울":
        return None
    normalized = _normalize(value).replace(" ", "")
    for district, aliases in SEOUL_DISTRICT_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return district
    direct = re.search(r"([가-힣]+구)", normalized)
    return direct.group(1) if direct else None


def _normalize(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(re.findall(r"[가-힣a-z0-9]+", value.lower()))


def _tokens(value: str | None) -> set[str]:
    normalized = _normalize(value)
    tokens = set(normalized.split())
    # '검은색카드지갑'처럼 붙어 있는 한국어도 부분 일치를 계산할 수 있게 2글자 조각을 추가한다.
    compact = normalized.replace(" ", "")
    tokens.update(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
    return {token for token in tokens if token}


def _text_similarity(left: str | None, right: str | None) -> float:
    a, b = _normalize(left), _normalize(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        containment = min(len(a), len(b)) / max(len(a), len(b))
        return min(1.0, 0.75 + 0.25 * containment)
    token_a, token_b = _tokens(a), _tokens(b)
    jaccard = len(token_a & token_b) / len(token_a | token_b) if token_a | token_b else 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    return max(jaccard, sequence * 0.9)


def _canonical_color(value: str | None) -> str | None:
    normalized = _normalize(value)
    for canonical, aliases in COLOR_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return canonical
    return normalized or None


def _color_similarity(left: str | None, right: str | None) -> float:
    left_color = _canonical_color(left)
    right_color = _canonical_color(right)
    if not left_color or not right_color:
        return 0.0
    return 1.0 if left_color == right_color else _text_similarity(left_color, right_color) * 0.4


def _date_similarity(lost_date: date | None, found_date: date | None) -> float:
    if not lost_date or not found_date:
        return 0.0
    difference = (found_date - lost_date).days
    if difference < 0:
        return 0.2 if difference >= -1 else 0.0
    if difference <= 1:
        return 1.0
    if difference <= 3:
        return 0.85
    if difference <= 7:
        return 0.65
    if difference <= 14:
        return 0.4
    if difference <= 30:
        return 0.15
    return 0.0


def _feature_similarity(features: Sequence[str], record_text: str) -> float:
    if not features:
        return 0.0
    scores = [_text_similarity(feature, record_text) for feature in features]
    return sum(scores) / len(scores)


def _confidence_level(
    query: LostItemQuery,
    score: float,
    scores: Mapping[str, float],
    location_scope: str,
) -> str:
    if score >= 75:
        confidence = "high"
    elif score >= 55:
        confidence = "medium"
    else:
        confidence = "low"

    if location_scope in {"nationwide", "adjacent"}:
        return "low"
    if location_scope in {"nearby", "region"} and confidence == "high":
        return "medium"
    if (query.lost_place or query.region) and scores["place"] < 0.3:
        return "medium" if confidence == "high" else confidence
    return confidence


def _location_scope_reason(location_scope: str) -> str:
    return {
        "direct": "입력한 장소와 직접 일치하는 범위",
        "district": "같은 시군구 범위",
        "nearby": "인접 시군구까지 확대한 범위",
        "region": "같은 시도 전체로 확대한 범위",
        "adjacent": "인접 시도까지 확대한 범위",
        "nationwide": "전국으로 확대한 낮은 신뢰도 범위",
    }[location_scope]


def _build_reasons(
    query: LostItemQuery,
    record: SearchRecord,
    scores: Mapping[str, float],
    vision_score: float | None,
) -> list[str]:
    reasons: list[str] = []
    labels = {
        "name": "물품명",
        "category": "분류",
        "color": "색상",
        "place": "장소",
        "date": "날짜",
        "features": "특징",
    }
    for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        if value >= 0.7 and _query_has_value(query, key):
            reasons.append(f"{labels[key]} 유사도가 높음({round(value * 100)}%)")
    if vision_score is not None and vision_score >= 0.7:
        reasons.append(f"사진 유사도가 높음({round(vision_score * 100)}%)")
    if record.image_url:
        reasons.append("확인 가능한 습득물 사진이 있음")
    if (query.lost_place or query.region) and scores["place"] < 0.3:
        reasons.append("입력한 장소와 일치 여부를 확인하기 어려움")
    return reasons or ["물품명 중심으로 검색된 후보이며 추가 확인이 필요함"]
