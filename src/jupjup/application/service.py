"""JupJup Agent가 사용하는 조회 사용 사례의 공개 파사드."""

from __future__ import annotations

from ..domain.matcher import LostItemMatcher
from ..domain.models import AgentResult, LostItemQuery
from .found_search import FoundItemSearch
from .lost_reports import SimilarLostReportSearch
from .ports import LostItemSearchPort, VisionMatcherPort


class JupJupAgentService:
    """Agent Tool에 안정적인 API를 제공하고 세부 사용 사례에 위임한다."""

    def __init__(
        self,
        api_client: LostItemSearchPort,
        matcher: LostItemMatcher | None = None,
        vision_matcher: VisionMatcherPort | None = None,
    ) -> None:
        self.api_client = api_client
        self.matcher = matcher or LostItemMatcher()
        self.vision_matcher = vision_matcher
        self.found_search = FoundItemSearch(
            api_client,
            self.matcher,
            vision_matcher,
        )
        self.similar_report_search = SimilarLostReportSearch(
            api_client,
            self.matcher,
        )

    def run(
        self,
        query: LostItemQuery,
        *,
        candidate_limit: int = 5,
        vision_limit: int = 3,
    ) -> AgentResult:
        return self.found_search.run(
            query,
            candidate_limit=candidate_limit,
            vision_limit=vision_limit,
        )

    def find_similar_lost_reports(
        self,
        query: LostItemQuery,
        *,
        report_limit: int = 5,
    ) -> AgentResult:
        return self.similar_report_search.run(query, report_limit=report_limit)
