from __future__ import annotations

import unittest
from datetime import date
from typing import Collection

from jupjup.models import ApiSearchResponse, LostItemQuery, RecordSource, SearchRecord
from jupjup.service import JupJupAgentService


class PartialApiClient:
    def __init__(self) -> None:
        self.requested_sources: list[set[RecordSource]] = []

    def search_all(
        self,
        query: LostItemQuery,
        *,
        sources: Collection[RecordSource] | None = None,
    ):
        selected_sources = set(sources or RecordSource)
        self.requested_sources.append(selected_sources)
        lost_report = SearchRecord(
            source=RecordSource.POLICE_LOST,
            record_type="lost",
            atc_id="L1",
            item_name="검정 카드지갑",
            event_date=date(2026, 9, 8),
            event_place="강남역",
        )
        newer_dissimilar_report = SearchRecord(
            source=RecordSource.POLICE_LOST,
            record_type="lost",
            atc_id="L2",
            item_name="하얀 여행가방",
            event_date=date(2026, 9, 10),
            event_place="제주공항",
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
                source=RecordSource.POLICE_LOST,
                total_count=2,
                records=[lost_report, newer_dissimilar_report],
            ),
            ApiSearchResponse(
                source=RecordSource.POLICE_FOUND, total_count=1, records=[found_item]
            ),
        ]
        responses = [
            response for response in responses if response.source in selected_sources
        ]
        errors = (
            {"포털기관 습득물": "테스트 오류"}
            if RecordSource.PORTAL_FOUND in selected_sources
            else {}
        )
        return responses, errors


class FailingVisionMatcher:
    def score(self, query: LostItemQuery, record: SearchRecord):
        raise RuntimeError("이미지 모델 일시 장애")


class ServiceTest(unittest.TestCase):
    def test_partial_api_failure_keeps_available_results(self) -> None:
        query = LostItemQuery(item_name="지갑", lost_date=date(2026, 9, 9), search_ready=True)
        api_client = PartialApiClient()
        service = JupJupAgentService(api_client)  # type: ignore[arg-type]

        result = service.run(query)

        self.assertEqual(result.candidates[0].record.atc_id, "F1")
        self.assertEqual(result.similar_lost_reports, [])
        self.assertEqual(
            api_client.requested_sources[-1],
            {RecordSource.POLICE_FOUND, RecordSource.PORTAL_FOUND},
        )
        self.assertIn("포털기관 습득물", result.errors)

    def test_similar_lost_reports_use_only_police_lost_source(self) -> None:
        query = LostItemQuery(
            item_name="카드지갑",
            lost_date=date(2026, 9, 9),
            lost_place="강남역",
            region="서울",
            search_ready=True,
        )
        api_client = PartialApiClient()
        service = JupJupAgentService(api_client)  # type: ignore[arg-type]

        result = service.find_similar_lost_reports(query)

        self.assertEqual(result.candidates, [])
        self.assertEqual(result.similar_lost_reports[0].atc_id, "L1")
        self.assertEqual(
            api_client.requested_sources[-1],
            {RecordSource.POLICE_LOST},
        )

    def test_vision_failure_keeps_text_candidates(self) -> None:
        query = LostItemQuery(item_name="지갑", search_ready=True)
        service = JupJupAgentService(
            PartialApiClient(),  # type: ignore[arg-type]
            vision_matcher=FailingVisionMatcher(),  # type: ignore[arg-type]
        )

        result = service.run(query)

        self.assertEqual(result.candidates[0].record.atc_id, "F1")
        self.assertEqual(
            result.errors["이미지 비교"],
            "일부 이미지 판정 실패 (RuntimeError)",
        )


if __name__ == "__main__":
    unittest.main()
