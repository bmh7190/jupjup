from datetime import date
import unittest

from jupjup.agent import JupJupChatResponse
from jupjup.demo import run_demo
from jupjup.models import (
    RecordSource,
    SearchRecord,
    SearchScope,
)
from jupjup.web import _response_for_web_chat


class WebChatResponseTest(unittest.TestCase):
    def test_hides_internal_search_diagnostics(self) -> None:
        result = run_demo().model_copy(
            update={
                "similar_lost_reports": [
                    SearchRecord(
                        source=RecordSource.POLICE_LOST,
                        record_type="lost",
                        atc_id="lost-1",
                        item_name="카드지갑",
                    )
                ],
                "search_scopes": [
                    SearchScope(
                        source=RecordSource.POLICE_LOST,
                        start_date=date(2026, 9, 1),
                        end_date=date(2026, 9, 7),
                        partial_reason="설정된 페이지 상한에 도달",
                    )
                ],
            }
        )
        original = JupJupChatResponse(message="검색 완료", search_result=result)

        filtered = _response_for_web_chat(original)
        payload = filtered.model_dump(mode="json")

        self.assertIsNotNone(filtered.search_result)
        public_result = payload["search_result"]
        self.assertNotIn("search_scopes", public_result)
        self.assertNotIn("similar_lost_reports", public_result)
        candidate = public_result["candidates"][0]
        self.assertEqual(set(candidate), {"record"})
        self.assertNotIn("score", candidate)
        self.assertNotIn("confidence", candidate)
        self.assertNotIn("reasons", candidate)
        self.assertEqual(candidate["record"]["category"], "지갑 > 카드지갑")
        self.assertEqual(candidate["record"]["color"], "블랙(검정)")
        self.assertEqual(candidate["record"]["event_time"], "20:10")
        self.assertEqual(candidate["record"]["status"], "보관중")
        self.assertNotIn("telephone", candidate["record"])
        self.assertNotIn("description", candidate["record"])
        self.assertIsNotNone(original.search_result)
        self.assertTrue(original.search_result.search_scopes)  # type: ignore[union-attr]
        self.assertTrue(original.search_result.similar_lost_reports)  # type: ignore[union-attr]

    def test_preserves_non_search_response(self) -> None:
        original = JupJupChatResponse(message="어떤 물품을 찾으시나요?")

        filtered = _response_for_web_chat(original)

        self.assertEqual(filtered.message, original.message)
        self.assertIsNone(filtered.search_result)
        self.assertIsNone(filtered.report_draft)


if __name__ == "__main__":
    unittest.main()
