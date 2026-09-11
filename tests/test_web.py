from datetime import date
import unittest

from jupjup.agent import JupJupChatResponse
from jupjup.models import (
    AgentResult,
    LostItemQuery,
    RecordSource,
    SearchRecord,
    SearchScope,
)
from jupjup.web import _response_for_web_chat


class WebChatResponseTest(unittest.TestCase):
    def test_hides_internal_search_diagnostics(self) -> None:
        result = AgentResult(
            query=LostItemQuery(item_name="지갑", search_ready=True),
            candidates=[],
            similar_lost_reports=[
                SearchRecord(
                    source=RecordSource.POLICE_LOST,
                    record_type="lost",
                    atc_id="lost-1",
                    item_name="카드지갑",
                )
            ],
            source_counts={"경찰청 분실물": 1},
            search_scopes=[
                SearchScope(
                    source=RecordSource.POLICE_LOST,
                    start_date=date(2026, 9, 1),
                    end_date=date(2026, 9, 7),
                    partial_reason="설정된 페이지 상한에 도달",
                )
            ],
        )
        original = JupJupChatResponse(message="검색 완료", search_result=result)

        filtered = _response_for_web_chat(original)

        self.assertIsNotNone(filtered.search_result)
        self.assertEqual(filtered.search_result.search_scopes, [])  # type: ignore[union-attr]
        self.assertEqual(  # type: ignore[union-attr]
            filtered.search_result.similar_lost_reports, []
        )
        self.assertIsNotNone(original.search_result)
        self.assertTrue(original.search_result.search_scopes)  # type: ignore[union-attr]
        self.assertTrue(original.search_result.similar_lost_reports)  # type: ignore[union-attr]

    def test_preserves_non_search_response(self) -> None:
        original = JupJupChatResponse(message="어떤 물품을 찾으시나요?")

        self.assertIs(_response_for_web_chat(original), original)


if __name__ == "__main__":
    unittest.main()
