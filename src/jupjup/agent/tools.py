"""모델이 호출할 수 있는 JupJup 도구 정의."""

from __future__ import annotations

import json
from datetime import date

from langchain_core.tools import BaseTool, tool

from ..application.service import JupJupAgentService
from ..domain.models import AgentResult, LostItemQuery, LostReportDraft
from ..domain.report import build_lost_report_draft
def _result_for_model(result: AgentResult) -> str:
    """LLM에는 사용자 안내에 필요한 필드만 전달해 컨텍스트를 제한한다."""
    payload = {
        "query": result.query.model_dump(mode="json", exclude_none=True),
        "source_counts": result.source_counts,
        "errors": result.errors,
        "search_scopes": [scope.model_dump(mode="json") for scope in result.search_scopes],
        "candidates": [
            {
                "source": candidate.record.source.label,
                "item_name": candidate.record.item_name,
                "category": candidate.record.category,
                "color": candidate.record.color,
                "event_date": candidate.record.event_date,
                "event_time": candidate.record.event_time,
                "event_place": candidate.record.event_place,
                "custody_place": candidate.record.custody_place,
                "status": candidate.record.status,
                "organization_name": candidate.record.organization_name,
                "image_url": candidate.record.image_url,
                "detail_url": candidate.record.detail_url,
            }
            for candidate in result.candidates
        ],
        "similar_lost_reports": [
            {
                "item_name": record.item_name,
                "event_date": record.event_date,
                "event_place": record.event_place,
                "detail_url": record.detail_url,
            }
            for record in result.similar_lost_reports
        ],
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def create_lost112_search_tool(service: JupJupAgentService) -> BaseTool:
    """도메인 서비스를 모델이 선택해 호출할 수 있는 Tool로 감싼다."""

    @tool("search_lost112_candidates", response_format="content_and_artifact")
    def search_lost112_candidates(
        item_name: str,
        category: str | None = None,
        lost_date: date | None = None,
        lost_time: str | None = None,
        lost_place: str | None = None,
        region: str | None = None,
        color: str | None = None,
        brand: str | None = None,
        features: list[str] | None = None,
        candidate_limit: int = 5,
    ) -> tuple[str, AgentResult]:
        """경찰청·포털기관 습득물 API를 조회하고 유사 후보를 추천한다.

        사용자가 잃어버린 물건을 설명했고 최소한 물품명을 알 수 있을 때 호출한다.
        제공되지 않은 조건은 빈 값으로 둔다.
        """
        query = LostItemQuery(
            item_name=item_name,
            category=category,
            lost_date=lost_date,
            lost_time=lost_time,
            lost_place=lost_place,
            region=region,
            color=color,
            brand=brand,
            features=features or [],
            search_ready=True,
        )
        result = service.run(query, candidate_limit=max(1, min(candidate_limit, 10)))
        return _result_for_model(result), result

    return search_lost112_candidates


def create_similar_lost_reports_tool(service: JupJupAgentService) -> BaseTool:
    """다른 사용자의 분실 신고만 별도로 조회하는 Tool을 만든다."""

    @tool("search_similar_lost_reports", response_format="content_and_artifact")
    def search_similar_lost_reports(
        item_name: str,
        category: str | None = None,
        lost_date: date | None = None,
        lost_time: str | None = None,
        lost_place: str | None = None,
        region: str | None = None,
        color: str | None = None,
        brand: str | None = None,
        features: list[str] | None = None,
        report_limit: int = 5,
    ) -> tuple[str, AgentResult]:
        """경찰청에 등록된 다른 사용자의 유사 분실 신고를 조회한다.

        사용자가 유사 분실 신고 확인을 명시적으로 요청했을 때만 호출한다.
        같은 대화에서 이미 확인한 분실 조건은 Middleware가 합쳐준다.
        """
        query = LostItemQuery(
            item_name=item_name,
            category=category,
            lost_date=lost_date,
            lost_time=lost_time,
            lost_place=lost_place,
            region=region,
            color=color,
            brand=brand,
            features=features or [],
            search_ready=True,
        )
        result = service.find_similar_lost_reports(
            query,
            report_limit=max(1, min(report_limit, 10)),
        )
        return _result_for_model(result), result

    return search_similar_lost_reports


def create_lost_report_tool() -> BaseTool:
    """분실신고 내용을 점검하고 사용자가 검토할 초안을 만드는 Tool."""

    @tool("prepare_lost_report_draft", response_format="content_and_artifact")
    def prepare_lost_report_draft(
        item_name: str | None = None,
        category: str | None = None,
        lost_date: date | None = None,
        lost_time: str | None = None,
        lost_place: str | None = None,
        region: str | None = None,
        color: str | None = None,
        size: str | None = None,
        brand: str | None = None,
        quantity: int | None = None,
        features: list[str] | None = None,
        circumstances: str | None = None,
        incident_type: str = "분실",
    ) -> tuple[str, LostReportDraft]:
        """분실신고 초안, 누락 항목, 개선 방법과 공식 접수 링크를 반환한다.

        신고를 제출하거나 경찰민원24 계정에 접근하지 않는다. 대화에서 확인된
        사실만 전달하고, 모르는 값은 비워 둔다.
        """
        draft = build_lost_report_draft(
            item_name=item_name,
            category=category,
            lost_date=lost_date,
            lost_time=lost_time,
            lost_place=lost_place,
            region=region,
            color=color,
            size=size,
            brand=brand,
            quantity=quantity,
            features=features,
            circumstances=circumstances,
            incident_type=incident_type,
        )
        return draft.model_dump_json(exclude_none=True), draft

    return prepare_lost_report_draft
