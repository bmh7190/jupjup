"""습득물 후보 조회 사용 사례."""

from __future__ import annotations

from ..domain.matcher import LostItemMatcher
from ..domain.models import AgentResult, ApiSearchResponse, LostItemQuery, RecordSource
from .ports import LostItemSearchPort, VisionMatcherPort

FOUND_RECORD_SOURCES = frozenset(
    {RecordSource.POLICE_FOUND, RecordSource.PORTAL_FOUND}
)


class FoundItemSearch:
    """두 습득물 출처를 조회하고 텍스트·이미지 기준으로 후보를 정렬한다."""

    def __init__(
        self,
        api_client: LostItemSearchPort,
        matcher: LostItemMatcher,
        vision_matcher: VisionMatcherPort | None = None,
    ) -> None:
        self.api_client = api_client
        self.matcher = matcher
        self.vision_matcher = vision_matcher

    def run(
        self,
        query: LostItemQuery,
        *,
        candidate_limit: int = 5,
        vision_limit: int = 3,
    ) -> AgentResult:
        responses, errors = self.api_client.search_all(
            query,
            sources=FOUND_RECORD_SOURCES,
        )
        found_records = [
            record
            for response in responses
            for record in response.records
            if record.record_type == "found"
        ]

        vision_scores: dict[str, float] = {}
        if self.vision_matcher:
            pre_ranked = self.matcher.rank(query, found_records, limit=vision_limit)
            for candidate in pre_ranked:
                try:
                    assessment = self.vision_matcher.score(query, candidate.record)
                except Exception as exc:
                    errors.setdefault(
                        "이미지 비교",
                        f"일부 이미지 판정 실패 ({type(exc).__name__})",
                    )
                    continue
                if assessment:
                    vision_scores[candidate.record.atc_id] = assessment.similarity

        candidates = self.matcher.rank(
            query,
            found_records,
            limit=candidate_limit,
            vision_scores=vision_scores,
        )
        return AgentResult(
            query=query,
            candidates=candidates,
            similar_lost_reports=[],
            source_counts=successful_source_counts(responses),
            errors=errors,
            search_scopes=[
                scope for response in responses for scope in response.search_scopes
            ],
        )


def successful_source_counts(
    responses: list[ApiSearchResponse],
) -> dict[str, int]:
    """실제로 한 페이지 이상 받은 출처만 성공 건수로 집계한다."""
    return {
        response.source.label: response.total_count
        for response in responses
        if not response.search_scopes
        or any(scope.pages_completed > 0 for scope in response.search_scopes)
    }
