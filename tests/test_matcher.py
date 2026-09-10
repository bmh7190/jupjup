from __future__ import annotations

import unittest
from datetime import date

from lost112_agent.matcher import LostItemMatcher
from lost112_agent.models import LostItemQuery, RecordSource, SearchRecord


class MatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query = LostItemQuery(
            item_name="지갑",
            category="카드지갑",
            lost_date=date(2026, 9, 9),
            lost_place="강남역",
            color="검정",
            features=["모서리 흠집"],
            search_ready=True,
        )

    def test_matching_candidate_ranks_first(self) -> None:
        close = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="close",
            item_name="검정색 카드지갑",
            category="지갑 > 카드지갑",
            event_date=date(2026, 9, 10),
            event_place="강남역 10번 출구",
            color="블랙",
            description="모서리에 흠집이 있음",
        )
        far = SearchRecord(
            source=RecordSource.PORTAL_FOUND,
            record_type="found",
            atc_id="far",
            item_name="갈색 장지갑",
            category="지갑 > 장지갑",
            event_date=date(2026, 8, 1),
            event_place="부산역",
            color="갈색",
        )

        ranked = LostItemMatcher().rank(self.query, [far, close])

        self.assertEqual(ranked[0].record.atc_id, "close")
        self.assertGreater(ranked[0].score, ranked[1].score)
        self.assertTrue(any("날짜" in reason for reason in ranked[0].reasons))

    def test_lost_report_is_not_returned_as_found_candidate(self) -> None:
        lost = SearchRecord(
            source=RecordSource.POLICE_LOST,
            record_type="lost",
            atc_id="lost",
            item_name="검정 카드지갑",
        )
        self.assertEqual(LostItemMatcher().rank(self.query, [lost]), [])


if __name__ == "__main__":
    unittest.main()

