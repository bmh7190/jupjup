"""Agent 단계 사이에서 주고받는 데이터 모델."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RecordSource(str, Enum):
    POLICE_LOST = "police_lost"
    POLICE_FOUND = "police_found"
    PORTAL_FOUND = "portal_found"

    @property
    def label(self) -> str:
        return {
            self.POLICE_LOST: "경찰청 분실물",
            self.POLICE_FOUND: "경찰청 습득물",
            self.PORTAL_FOUND: "포털기관 습득물",
        }[self]


class LostItemQuery(BaseModel):
    """사용자 대화에서 추출한 분실물 검색 조건."""

    item_name: str | None = Field(default=None, description="분실한 물품의 일반적인 이름")
    category: str | None = Field(default=None, description="지갑, 가방, 휴대폰 같은 물품 분류")
    lost_date: date | None = Field(default=None, description="분실일 YYYY-MM-DD")
    lost_time: str | None = Field(default=None, description="분실 시각 HH:MM 또는 시간대")
    lost_place: str | None = Field(default=None, description="구체적인 분실 장소")
    region: str | None = Field(default=None, description="시도 또는 시군구")
    color: str | None = Field(default=None, description="대표 색상")
    brand: str | None = Field(default=None, description="브랜드 또는 제조사")
    features: list[str] = Field(default_factory=list, description="모양, 내용물, 흠집 등 식별 특징")
    missing_fields: list[str] = Field(default_factory=list, description="추가로 확인해야 할 필드명")
    follow_up_question: str | None = Field(default=None, description="사용자에게 할 짧은 추가 질문")
    search_ready: bool = Field(default=False, description="검색에 필요한 최소 정보가 모였는지 여부")

    @field_validator("item_name", "category", "lost_time", "lost_place", "region", "color", "brand")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SearchRecord(BaseModel):
    """세 API의 서로 다른 XML 필드를 동일한 형태로 변환한 레코드."""

    source: RecordSource
    record_type: str = Field(pattern="^(lost|found)$")
    atc_id: str
    sequence: str | None = None
    item_name: str = ""
    category: str | None = None
    event_date: date | None = None
    event_time: str | None = None
    event_place: str | None = None
    custody_place: str | None = None
    color: str | None = None
    subject: str | None = None
    description: str | None = None
    image_url: str | None = None
    organization_name: str | None = None
    telephone: str | None = None
    status: str | None = None
    detail_url: str | None = None

    @property
    def searchable_text(self) -> str:
        values = [self.item_name, self.category, self.color, self.subject, self.description]
        return " ".join(value for value in values if value)


class ScoreBreakdown(BaseModel):
    name: float = 0.0
    category: float = 0.0
    color: float = 0.0
    place: float = 0.0
    date: float = 0.0
    features: float = 0.0
    image: float | None = None


class MatchCandidate(BaseModel):
    record: SearchRecord
    score: float = Field(ge=0.0, le=100.0)
    breakdown: ScoreBreakdown
    reasons: list[str] = Field(default_factory=list)


class ApiSearchResponse(BaseModel):
    source: RecordSource
    total_count: int
    records: list[SearchRecord]
    search_scopes: list[SearchScope] = Field(default_factory=list)


class SearchScope(BaseModel):
    source: RecordSource
    start_date: date
    end_date: date
    pages_completed: int = 0
    retrieved_count: int = 0
    total_count: int = 0
    complete: bool = False
    partial_reason: str | None = None
    error: str | None = None


class AgentResult(BaseModel):
    query: LostItemQuery
    candidates: list[MatchCandidate]
    similar_lost_reports: list[SearchRecord]
    source_counts: dict[str, int]
    errors: dict[str, str] = Field(default_factory=dict)
    search_scopes: list[SearchScope] = Field(default_factory=list)


class LostReportDraft(BaseModel):
    """경찰민원24에 옮겨 적기 전에 사용자가 검토하는 분실신고 초안."""

    title: str
    copy_text: str
    item_name: str | None = None
    category: str | None = None
    lost_date: date | None = None
    lost_time: str | None = None
    lost_place: str | None = None
    region: str | None = None
    color: str | None = None
    size: str | None = None
    brand: str | None = None
    quantity: int | None = Field(default=None, ge=1)
    features: list[str] = Field(default_factory=list)
    circumstances: str | None = None
    missing_essential_fields: list[str] = Field(default_factory=list)
    missing_recommended_fields: list[str] = Field(default_factory=list)
    improvement_tips: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    next_question: str | None = None
    ready_for_user_review: bool = False
    official_report_url: str
    official_guide_url: str
    auto_submitted: bool = False


class VisionAssessment(BaseModel):
    similarity: float = Field(ge=0.0, le=1.0, description="설명과 사진의 일치도")
    reason: str = Field(description="판정 근거 한 문장")
