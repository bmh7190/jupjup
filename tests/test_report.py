from __future__ import annotations

import unittest
from datetime import date

from jupjup.report import (
    POLICE_REPORT_GUIDE_URL,
    POLICE_REPORT_URL,
    build_lost_report_draft,
)


class LostReportDraftTest(unittest.TestCase):
    def test_complete_input_builds_copyable_draft(self) -> None:
        draft = build_lost_report_draft(
            item_name="카드지갑",
            category="지갑",
            lost_date=date(2026, 9, 10),
            lost_time="18:30경",
            lost_place="강남역 2호선 승강장",
            region="서울 강남구",
            color="검정",
            size="가로 11cm",
            brand="몽블랑",
            quantity=1,
            features=["흰색 스티치", "교통카드 1장"],
            circumstances="퇴근길 열차에서 내린 뒤 분실을 확인함",
        )

        self.assertTrue(draft.ready_for_user_review)
        self.assertEqual(draft.missing_essential_fields, [])
        self.assertIn("강남역 2호선 승강장", draft.copy_text)
        self.assertIn("흰색 스티치", draft.copy_text)
        self.assertIn("분실 수량: 1", draft.copy_text)
        self.assertIn("가로 11cm", draft.copy_text)
        self.assertEqual(draft.official_report_url, POLICE_REPORT_URL)
        self.assertEqual(draft.official_guide_url, POLICE_REPORT_GUIDE_URL)
        self.assertIn("cvlcptAply.do", draft.official_report_url)
        self.assertFalse(draft.auto_submitted)

    def test_missing_information_returns_question_and_improvement_tips(self) -> None:
        draft = build_lost_report_draft(item_name="지갑")

        self.assertFalse(draft.ready_for_user_review)
        self.assertIn("분실 날짜", draft.missing_essential_fields)
        self.assertIn("구체적인 분실 장소", draft.missing_essential_fields)
        self.assertIn("분실 수량", draft.missing_recommended_fields)
        self.assertIsNotNone(draft.next_question)
        self.assertIn("분실 날짜", draft.next_question)
        self.assertNotIn("구체적인 분실 장소", draft.next_question)
        self.assertTrue(draft.improvement_tips)

    def test_optional_information_does_not_create_follow_up_question(self) -> None:
        draft = build_lost_report_draft(
            item_name="지갑",
            lost_date=date(2026, 9, 10),
            lost_place="강남역",
        )

        self.assertTrue(draft.ready_for_user_review)
        self.assertTrue(draft.missing_recommended_fields)
        self.assertIsNone(draft.next_question)

    def test_pii_is_masked_from_copy_text(self) -> None:
        draft = build_lost_report_draft(
            item_name="지갑",
            lost_date=date(2026, 9, 10),
            lost_place="강남역",
            features=["카드번호 1234-5678-9012-3456"],
            circumstances="연락처는 010-1234-5678",
        )

        self.assertNotIn("1234-5678-9012-3456", draft.copy_text)
        self.assertNotIn("010-1234-5678", draft.copy_text)
        self.assertIn("[카드번호 마스킹]", draft.copy_text)
        self.assertIn("[전화번호 마스킹]", draft.copy_text)

    def test_theft_and_vehicle_plate_have_special_guidance(self) -> None:
        theft = build_lost_report_draft(
            item_name="가방",
            lost_date=date(2026, 9, 10),
            lost_place="강남역",
            incident_type="도난",
        )
        plate = build_lost_report_draft(
            item_name="자동차번호판",
            lost_date=date(2026, 9, 10),
            lost_place="주차장",
        )

        self.assertFalse(theft.ready_for_user_review)
        self.assertTrue(any("도난" in notice for notice in theft.notices))
        self.assertTrue(any("방문 신고" in notice for notice in plate.notices))


if __name__ == "__main__":
    unittest.main()
