"""추출, 조회, 이미지 판정, 후보 추천을 순서대로 조정한다."""

from __future__ import annotations

from datetime import date

from .api_client import Lost112ApiClient
from .matcher import LostItemMatcher
from .models import AgentResult, LostItemQuery, RecordSource
from .vision import VisionMatcher


class Lost112AgentService:
    def __init__(
        self,
        api_client: Lost112ApiClient,
        matcher: LostItemMatcher | None = None,
        vision_matcher: VisionMatcher | None = None,
    ) -> None:
        self.api_client = api_client
        self.matcher = matcher or LostItemMatcher()
        self.vision_matcher = vision_matcher

    def run(
        self,
        query: LostItemQuery,
        *,
        candidate_limit: int = 5,
        vision_limit: int = 3,
    ) -> AgentResult:
        responses, errors = self.api_client.search_all(query)
        records = [record for response in responses for record in response.records]
        found_records = [record for record in records if record.record_type == "found"]
        lost_records = [record for record in records if record.record_type == "lost"]

        vision_scores: dict[str, float] = {}
        if self.vision_matcher:
            # 텍스트 점수가 높은 소수 후보에만 비전 모델을 적용해 비용과 시간을 제한한다.
            pre_ranked = self.matcher.rank(query, found_records, limit=vision_limit)
            for candidate in pre_ranked:
                assessment = self.vision_matcher.score(query, candidate.record)
                if assessment:
                    vision_scores[candidate.record.atc_id] = assessment.similarity

        candidates = self.matcher.rank(
            query, found_records, limit=candidate_limit, vision_scores=vision_scores
        )
        source_counts = {
            response.source.label: response.total_count for response in responses
        }
        # 동일 물품명으로 접수된 다른 분실 신고는 후보와 분리해서 참고 정보로 제공한다.
        similar_lost_reports = sorted(
            lost_records,
            key=lambda record: record.event_date or date.min,
            reverse=True,
        )[:candidate_limit]

        return AgentResult(
            query=query,
            candidates=candidates,
            similar_lost_reports=similar_lost_reports,
            source_counts=source_counts,
            errors=errors,
        )
