"""추출, 조회, 이미지 판정, 후보 추천을 순서대로 조정한다."""

from __future__ import annotations

from datetime import date

from .api_client import FOUND_RECORD_SOURCES, Lost112ApiClient
from .matcher import LostItemMatcher
from .models import AgentResult, LostItemQuery, RecordSource
from .vision import VisionMatcher


class JupJupAgentService:
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
        responses, errors = self.api_client.search_all(
            query,
            sources=FOUND_RECORD_SOURCES,
        )
        records = [record for response in responses for record in response.records]
        found_records = [record for record in records if record.record_type == "found"]

        vision_scores: dict[str, float] = {}
        if self.vision_matcher:
            # 텍스트 점수가 높은 소수 후보에만 비전 모델을 적용해 비용과 시간을 제한한다.
            pre_ranked = self.matcher.rank(query, found_records, limit=vision_limit)
            for candidate in pre_ranked:
                try:
                    assessment = self.vision_matcher.score(query, candidate.record)
                except Exception as exc:
                    # 선택 기능의 실패가 공개 데이터 기반 텍스트 후보를 없애지 않게 한다.
                    errors.setdefault(
                        "이미지 비교",
                        f"일부 이미지 판정 실패 ({type(exc).__name__})",
                    )
                    continue
                if assessment:
                    vision_scores[candidate.record.atc_id] = assessment.similarity

        candidates = self.matcher.rank(
            query, found_records, limit=candidate_limit, vision_scores=vision_scores
        )
        source_counts = {
            response.source.label: response.total_count
            for response in responses
            # 날짜 검색은 실패한 호출도 오류 scope를 보존하기 위해 빈 응답으로
            # 돌려준다. 한 페이지도 받지 못한 출처를 성공한 0건으로 세지 않는다.
            if not response.search_scopes
            or any(scope.pages_completed > 0 for scope in response.search_scopes)
        }
        return AgentResult(
            query=query,
            candidates=candidates,
            similar_lost_reports=[],
            source_counts=source_counts,
            errors=errors,
            search_scopes=[
                scope for response in responses for scope in response.search_scopes
            ],
        )

    def find_similar_lost_reports(
        self,
        query: LostItemQuery,
        *,
        report_limit: int = 5,
    ) -> AgentResult:
        """사용자가 요청했을 때만 경찰청의 다른 분실 신고를 조회한다."""
        responses, errors = self.api_client.search_all(
            query,
            sources={RecordSource.POLICE_LOST},
        )
        lost_records = [
            record
            for response in responses
            for record in response.records
            if record.record_type == "lost"
        ]
        similar_lost_reports = sorted(
            lost_records,
            key=lambda record: (
                self.matcher.score(query, record).score,
                record.event_date or date.min,
            ),
            reverse=True,
        )[:report_limit]
        source_counts = {
            response.source.label: response.total_count
            for response in responses
            if not response.search_scopes
            or any(scope.pages_completed > 0 for scope in response.search_scopes)
        }
        return AgentResult(
            query=query,
            candidates=[],
            similar_lost_reports=similar_lost_reports,
            source_counts=source_counts,
            errors=errors,
            search_scopes=[
                scope for response in responses for scope in response.search_scopes
            ],
        )
