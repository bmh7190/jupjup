from __future__ import annotations

import unittest
from datetime import date

from lost112_agent.models import ApiSearchResponse, LostItemQuery, RecordSource, SearchRecord
from lost112_agent.service import Lost112AgentService


class PartialApiClient:
    def search_all(self, query: LostItemQuery):
        lost_report = SearchRecord(
            source=RecordSource.POLICE_LOST,
            record_type="lost",
            atc_id="L1",
            item_name="검정 카드지갑",
            event_date=date(2026, 9, 8),
        )
        found_item = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="F1",
            item_name="검정 카드지갑",
            event_date=date(2026, 9, 9),
        )
        responses = [
            ApiSearchResponse(
                source=RecordSource.POLICE_LOST, total_count=1, records=[lost_report]
            ),
            ApiSearchResponse(
                source=RecordSource.POLICE_FOUND, total_count=1, records=[found_item]
            ),
        ]
        return responses, {"포털기관 습득물": "테스트 오류"}


class ServiceTest(unittest.TestCase):
    def test_partial_api_failure_keeps_available_results(self) -> None:
        query = LostItemQuery(item_name="지갑", lost_date=date(2026, 9, 9), search_ready=True)
        service = Lost112AgentService(PartialApiClient())  # type: ignore[arg-type]

        result = service.run(query)

        self.assertEqual(result.candidates[0].record.atc_id, "F1")
        self.assertEqual(result.similar_lost_reports[0].atc_id, "L1")
        self.assertIn("포털기관 습득물", result.errors)


if __name__ == "__main__":
    unittest.main()
