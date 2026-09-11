"""LangChain Agent와 LOST112 검색 Tool을 구성한다."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from .models import AgentResult, LostItemQuery
from .service import JupJupAgentService


SYSTEM_PROMPT = """당신은 분실물 찾기를 돕는 '줍줍이'입니다.
사용자와 한국어로 짧고 명확하게 대화하세요.

다음 원칙을 지키세요.
- 사용자의 설명에서 물품명, 분실일, 장소, 지역, 색상, 브랜드, 특징을 파악하세요.
- 물품명을 알 수 없으면 검색하지 말고 먼저 물어보세요.
- 물품명을 알 수 있으면 search_lost112_candidates Tool로 실제 데이터를 조회하세요.
- 사용자가 제공하지 않은 조건은 추측해서 Tool 인자에 넣지 마세요.
- 조회하지 않은 결과를 찾았다고 말하지 마세요.
- 습득물 후보와 다른 사람이 등록한 유사 분실 신고를 구분하세요.
- 후보를 안내할 때 점수, 일치 근거, 사진 URL, 상세 URL을 생략하지 마세요.
- API 오류가 있으면 성공한 출처와 실패한 출처를 구분해서 알려주세요.
- 주민등록번호, 카드번호, 전화번호, 이메일 등 개인정보를 답변에 노출하지 마세요.
"""


class JupJupChatResponse(BaseModel):
    """한 번의 Agent 실행 결과."""

    message: str
    search_result: AgentResult | None = None


def _result_for_model(result: AgentResult) -> str:
    """LLM에는 사용자 안내에 필요한 필드만 전달해 컨텍스트를 제한한다."""
    payload = {
        "query": result.query.model_dump(mode="json", exclude_none=True),
        "source_counts": result.source_counts,
        "errors": result.errors,
        "candidates": [
            {
                "score": candidate.score,
                "reasons": candidate.reasons,
                "source": candidate.record.source.label,
                "item_name": candidate.record.item_name,
                "event_date": candidate.record.event_date,
                "event_place": candidate.record.event_place,
                "custody_place": candidate.record.custody_place,
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
        """경찰청 분실물·습득물·포털기관 습득물 API를 조회하고 유사 후보를 추천한다.

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


def build_middlewares() -> list[Any]:
    """입출력 개인정보 보호와 Agent/Tool 과호출 방지 정책."""
    return [
        PIIMiddleware(
            "email",
            detector=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
        ),
        PIIMiddleware(
            "credit_card", strategy="mask", apply_to_input=True, apply_to_output=True
        ),
        PIIMiddleware(
            "phone_number",
            detector=r"(?<!\d)01[016789]-?\d{3,4}-?\d{4}(?!\d)",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
        ),
        PIIMiddleware(
            "resident_registration_number",
            detector=r"(?<!\d)\d{6}-?[1-8]\d{6}(?!\d)",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
        ),
        ToolRetryMiddleware(
            max_retries=1,
            tools=["search_lost112_candidates"],
            on_failure="continue",
            initial_delay=0.5,
        ),
        ToolCallLimitMiddleware(
            tool_name="search_lost112_candidates", run_limit=1
        ),
        ModelCallLimitMiddleware(run_limit=4, exit_behavior="end"),
    ]


def _message_text(message: AIMessage | None) -> str:
    if message is None:
        return "답변을 생성하지 못했습니다."
    if isinstance(message.content, str):
        return message.content

    chunks: list[str] = []
    for block in message.content:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text", "")))
    return "\n".join(chunk for chunk in chunks if chunk) or "답변을 생성하지 못했습니다."


class JupJupChatAgent:
    """대화 Memory를 가진 실제 LangChain Tool-calling Agent."""

    def __init__(
        self,
        service: JupJupAgentService,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        if model is None:
            if not model_name or not api_key:
                raise ValueError("Agent 모델 생성에 model_name과 api_key가 필요합니다.")
            model = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)

        self.search_tool = create_lost112_search_tool(service)
        self.checkpointer = InMemorySaver()
        self.graph = create_agent(
            model=model,
            tools=[self.search_tool],
            system_prompt=SYSTEM_PROMPT,
            middleware=build_middlewares(),
            checkpointer=self.checkpointer,
            name="jupjup_agent",
        )

    def chat(self, text: str, *, thread_id: str) -> JupJupChatResponse:
        """thread_id별 대화를 기억하며 한 턴을 실행한다."""
        state = self.graph.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config={"configurable": {"thread_id": thread_id}},
        )

        final_message: AIMessage | None = None
        search_result: AgentResult | None = None
        for message in state["messages"]:
            if isinstance(message, AIMessage):
                final_message = message
            elif (
                isinstance(message, ToolMessage)
                and message.name == self.search_tool.name
                and isinstance(message.artifact, AgentResult)
            ):
                search_result = message.artifact

        return JupJupChatResponse(
            message=_message_text(final_message), search_result=search_result
        )
