"""웹 API가 외부에 공개하는 요청·응답 모델."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from ..domain.models import LostItemQuery, LostReportDraft, RecordSource, SearchRecord


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class WebSearchRecord(BaseModel):
    """습득물 후보에서 사용자 확인에 필요한 공개 필드."""

    source: RecordSource
    item_name: str
    category: str | None = None
    event_date: date | None = None
    event_time: str | None = None
    event_place: str | None = None
    custody_place: str | None = None
    color: str | None = None
    image_url: str | None = None
    organization_name: str | None = None
    status: str | None = None
    detail_url: str | None = None

    @classmethod
    def from_record(cls, record: SearchRecord) -> "WebSearchRecord":
        return cls.model_validate(record.model_dump())


class WebCandidate(BaseModel):
    """점수와 판정 근거를 제외한 사용자용 후보."""

    record: WebSearchRecord


class WebSearchResult(BaseModel):
    query: LostItemQuery
    candidates: list[WebCandidate]
    source_counts: dict[str, int]
    errors: dict[str, str]


class WebChatResponse(BaseModel):
    message: str
    search_result: WebSearchResult | None = None
    report_draft: LostReportDraft | None = None


class ChatReply(BaseModel):
    thread_id: str
    response: WebChatResponse
