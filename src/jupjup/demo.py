"""외부 API 키 없이 화면과 점수 계산을 확인하기 위한 데모 데이터."""

from __future__ import annotations

from datetime import date

from .matcher import LostItemMatcher
from .models import AgentResult, LostItemQuery, RecordSource, SearchRecord


def run_demo() -> AgentResult:
    query = LostItemQuery(
        item_name="지갑",
        category="카드지갑",
        lost_date=date(2026, 9, 9),
        lost_time="19:00",
        lost_place="강남역 10번 출구",
        region="서울 강남구",
        color="검정",
        brand="몽블랑",
        features=["얇은 카드지갑", "모서리 흠집"],
        search_ready=True,
    )
    records = [
        SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="F-DEMO-001",
            sequence="1",
            item_name="검정색 카드지갑",
            category="지갑 > 카드지갑",
            event_date=date(2026, 9, 9),
            event_time="20:10",
            event_place="강남역 10번 출구 인근",
            custody_place="서울강남경찰서",
            color="블랙(검정)",
            description="얇은 카드지갑이며 모서리에 작은 흠집이 있음",
            image_url="https://example.com/demo-wallet.jpg",
            organization_name="서울강남경찰서",
            status="보관중",
        ),
        SearchRecord(
            source=RecordSource.PORTAL_FOUND,
            record_type="found",
            atc_id="F-DEMO-002",
            sequence="1",
            item_name="갈색 남성용 반지갑",
            category="지갑 > 남성용 지갑",
            event_date=date(2026, 9, 12),
            event_time="14:30",
            event_place="서울역 대합실",
            custody_place="서울역 유실물센터",
            color="갈색",
            organization_name="서울역 유실물센터",
            status="반입중",
        ),
    ]
    candidates = LostItemMatcher().rank(query, records, limit=5)
    return AgentResult(
        query=query,
        candidates=candidates,
        similar_lost_reports=[],
        source_counts={"경찰청 습득물": 1, "포털기관 습득물": 1, "경찰청 분실물": 0},
    )
