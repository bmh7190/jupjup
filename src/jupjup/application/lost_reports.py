"""다른 사용자의 유사 분실 신고 조회 사용 사례."""

from __future__ import annotations

from datetime import date

from ..domain.matcher import LostItemMatcher
from ..domain.models import AgentResult, LostItemQuery, RecordSource
from .found_search import successful_source_counts
from .ports import LostItemSearchPort


class SimilarLostReportSearch:
    def __init__(
        self,
        api_client: LostItemSearchPort,
        matcher: LostItemMatcher,
    ) -> None:
        self.api_client = api_client
        self.matcher = matcher

    def run(self, query: LostItemQuery, *, report_limit: int = 5) -> AgentResult:
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
        return AgentResult(
            query=query,
            candidates=[],
            similar_lost_reports=similar_lost_reports,
            source_counts=successful_source_counts(responses),
            errors=errors,
            search_scopes=[
                scope for response in responses for scope in response.search_scopes
            ],
        )
