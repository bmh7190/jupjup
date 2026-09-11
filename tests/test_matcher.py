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
        self.assertEqual(ranked[0].confidence, "high")
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

    def test_unknown_candidate_region_is_nationwide_low_confidence(self) -> None:
        location_unknown = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="location-unknown",
            item_name="검정색 카드지갑",
            category="지갑 > 카드지갑",
            event_date=date(2026, 9, 10),
            custody_place="지역미상보관소",
            color="블랙",
            description="모서리에 흠집이 있음",
        )

        candidate = LostItemMatcher().rank(self.query, [location_unknown])[0]

        self.assertEqual(candidate.location_scope, "nationwide")
        self.assertEqual(candidate.confidence, "low")

    def test_candidate_in_distant_region_is_nationwide_low_confidence(self) -> None:
        busan = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="busan-wallet",
            item_name="검정색 카드지갑",
            event_date=date(2026, 9, 10),
            custody_place="좌동지구대",
            telephone="051-123-4567",
            color="블랙",
        )

        candidate = LostItemMatcher().rank(self.query, [busan])[0]

        self.assertEqual(candidate.score, 45.0)
        self.assertEqual(candidate.confidence, "low")
        self.assertEqual(candidate.location_scope, "nationwide")

    def test_candidate_in_same_broad_region_gets_location_score(self) -> None:
        seoul = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="seoul-wallet",
            item_name="검정색 카드지갑",
            category="지갑 > 카드지갑",
            event_date=date(2026, 9, 10),
            custody_place="상일파출소",
            telephone="02-1234-5678",
            color="블랙",
            description="모서리에 흠집이 있음",
        )

        candidate = LostItemMatcher().rank(self.query, [seoul])[0]

        self.assertEqual(candidate.breakdown.place, 0.4)
        self.assertEqual(candidate.confidence, "medium")
        self.assertEqual(candidate.location_scope, "region")

    def test_exact_metropolitan_region_name_is_recognized(self) -> None:
        query = self.query.model_copy(
            update={"lost_place": "광주 충장로", "region": "광주"}
        )
        gwangju = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="gwangju-wallet",
            item_name="검정색 카드지갑",
            event_date=date(2026, 9, 10),
            custody_place="광주동부경찰서",
            telephone="062-123-4567",
            color="블랙",
        )

        candidate = LostItemMatcher().rank(query, [gwangju])[0]

        self.assertEqual(candidate.location_scope, "region")

    def test_search_stops_at_nearest_location_tier(self) -> None:
        gangnam = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="gangnam-wallet",
            item_name="검정색 카드지갑",
            event_date=date(2026, 9, 10),
            custody_place="강남경찰서",
            telephone="02-1234-5678",
            color="블랙",
        )
        seoul = gangnam.model_copy(
            update={"atc_id": "seoul-wallet", "custody_place": "상일파출소"}
        )
        gyeonggi = gangnam.model_copy(
            update={
                "atc_id": "gyeonggi-wallet",
                "custody_place": "영통지구대",
                "telephone": "031-123-4567",
            }
        )

        ranked = LostItemMatcher().rank(self.query, [gyeonggi, seoul, gangnam])

        self.assertEqual(
            [candidate.record.atc_id for candidate in ranked], ["gangnam-wallet"]
        )
        self.assertEqual(ranked[0].location_scope, "district")

    def test_search_expands_to_gyeonggi_when_seoul_has_no_candidate(self) -> None:
        gyeonggi = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="gyeonggi-wallet",
            item_name="검정색 카드지갑",
            event_date=date(2026, 9, 10),
            custody_place="영통지구대",
            telephone="031-123-4567",
            color="블랙",
        )

        ranked = LostItemMatcher().rank(self.query, [gyeonggi])

        self.assertEqual(
            [candidate.record.atc_id for candidate in ranked], ["gyeonggi-wallet"]
        )
        self.assertEqual(ranked[0].location_scope, "adjacent")
        self.assertEqual(ranked[0].confidence, "low")

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
