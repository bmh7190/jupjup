from __future__ import annotations

import unittest
from datetime import date

from jupjup.matcher import LostItemMatcher
from jupjup.models import LostItemQuery, RecordSource, SearchRecord


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

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].record.atc_id, "close")
        self.assertTrue(any("날짜" in reason for reason in ranked[0].reasons))

    def test_item_found_before_loss_is_not_a_candidate(self) -> None:
        impossible = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="found-before-loss",
            item_name="검정 카드지갑",
            event_date=date(2026, 9, 8),
            event_place="강남역",
            color="검정",
        )

        self.assertEqual(LostItemMatcher().rank(self.query, [impossible]), [])

    def test_item_without_found_date_remains_a_candidate(self) -> None:
        unknown_date = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="unknown-date",
            item_name="검정 카드지갑",
            event_place="강남역",
            color="검정",
        )

        ranked = LostItemMatcher().rank(self.query, [unknown_date])

        self.assertEqual([candidate.record.atc_id for candidate in ranked], ["unknown-date"])

    def test_low_score_record_is_not_returned_as_candidate(self) -> None:
        unrelated = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="unrelated",
            item_name="분홍색 장난감 지갑",
            event_date=date(2026, 9, 20),
            event_place="제주공항",
            color="분홍",
        )

        self.assertEqual(LostItemMatcher().rank(self.query, [unrelated]), [])

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
