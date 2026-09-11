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


class LostItemMatcher:
    def rank(
        self,
        query: LostItemQuery,
        records: Sequence[SearchRecord],
        *,
        limit: int = 5,
        vision_scores: Mapping[str, float] | None = None,
    ) -> list[MatchCandidate]:
        candidates = [
            self.score(query, record, (vision_scores or {}).get(record.atc_id))
            for record in records
            if (
                record.record_type == "found"
                and _is_temporally_possible(query, record)
            )
        ]
        candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= MIN_CANDIDATE_SCORE
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[:limit]

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
        breakdown = ScoreBreakdown(**component_scores, image=vision_score)
        reasons = _build_reasons(query, record, component_scores, vision_score)
        return MatchCandidate(
            record=record,
            score=round(total * 100, 1),
            breakdown=breakdown,
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
    return reasons or ["물품명 중심으로 검색된 후보이며 추가 확인이 필요함"]
