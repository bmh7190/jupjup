"""LangChain Agent와 LOST112 검색 Tool을 구성한다."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from .models import AgentResult, LostItemQuery, LostReportDraft
from .report import build_lost_report_draft
from .service import JupJupAgentService


SYSTEM_PROMPT = """당신은 분실물 찾기를 돕는 '줍줍이'입니다.
사용자와 한국어로 짧고 명확하게 대화하세요.

다음 원칙을 지키세요.
- 먼저 사용자의 의도를 습득물 조회와 분실신고 작성 도움으로 구분하세요.
- 사용자의 설명에서 물품명, 분실일, 장소, 지역, 색상, 크기, 브랜드, 수량, 특징을 파악하세요.
- region은 광역 시도명으로 통일하세요. 예: 강남역·광진구는 서울, 우도는 제주입니다.
- item_name에는 '지갑'처럼 짧은 일반 물품명을 넣고 '샤넬' 같은 브랜드와 색상은 별도 인자로 전달하세요.
- 현재 메시지에서 사용자가 분실신고 작성, 문장 정리, 누락 확인을 명시적으로 요청한 경우에만 prepare_lost_report_draft Tool을 호출하세요.
- 단순히 물건을 잃어버렸다고 설명하거나 습득물 조회를 요청한 경우에는 신고서 Tool을 호출하지 마세요.
- 신고서 작성만 요청한 경우에는 search_lost112_candidates Tool을 호출하지 마세요.
- 신고서 작성 요청에서는 물품명, 분실 날짜, 구체적인 장소가 없으면 초안의 next_question으로 먼저 보완하세요.
- prepare_lost_report_draft 결과가 준비되면 복사용 문장, 누락 항목, 개선 제안, 주의사항을 안내하세요.
- 신고서 결과의 official_report_url과 official_guide_url을 답변에서 생략하지 마세요.
- 이 서비스는 신고를 자동 제출하지 않습니다. 제출됐다고 말하지 말고 경찰민원24 공식 링크를 안내하세요.
- 도난은 분실물 신고와 구분하고, 자동차번호판은 방문 신고가 필요하다는 Tool 결과를 따르세요.
- 조회 요청에서 물품명을 알 수 없으면 검색하지 말고 먼저 물어보세요.
- 조회 요청에서 물품명을 알 수 있으면 확인이나 동의를 다시 묻지 말고 같은 턴에 search_lost112_candidates Tool로 실제 데이터를 조회하세요.
- 사용자가 제공하지 않은 조건은 추측해서 Tool 인자에 넣지 마세요.
- 조회하지 않은 결과를 찾았다고 말하지 마세요.
- 습득물 후보와 다른 사람이 등록한 유사 분실 신고를 구분하세요.
- 후보를 안내할 때 점수, 일치 근거, 사진 URL, 상세 URL을 생략하지 마세요.
- API 오류가 있으면 성공한 출처와 실패한 출처를 구분해서 알려주세요.
- 날짜가 있으면 Tool이 분실일부터 7일, 다음 7일, 그 후 한 달 순서로 검색합니다. 기간을 임의로 최신 날짜로 바꾸지 마세요.
- search_scopes의 실제 조회 기간과 일부 조회 여부를 안내하세요. 조회 오류를 결과 없음으로 표현하지 마세요.
- 주민등록번호, 카드번호, 전화번호, 이메일 등 개인정보를 답변에 노출하지 마세요.
"""


class JupJupChatResponse(BaseModel):
    """한 번의 Agent 실행 결과."""

    message: str
    search_result: AgentResult | None = None
    report_draft: LostReportDraft | None = None


_REPORT_REQUEST_PATTERNS = (
    re.compile(
        r"(?:분실\s*)?신고(?:서|내용|문)?(?:를|을|도)?\s*"
        r".{0,12}(?:써|작성|정리|검토|점검|준비|만들|도와)"
    ),
    re.compile(
        r"(?:써|작성|정리|검토|점검|준비|만들|도와).{0,12}"
        r"(?:분실\s*)?신고(?:서|내용|문)?"
    ),
    re.compile(
        r"(?:민원\s*)?접수.{0,12}(?:도와|준비|작성|검토|점검|빠진|누락)"
    ),
    re.compile(r"(?:빠진|누락).{0,12}(?:내용|항목).{0,12}(?:봐|확인|검토)"),
)

_SEARCH_REQUEST_PATTERNS = (
    re.compile(r"(?:을|를)\s*(?:잃어버|분실(?:했|한|함)|두고|놓고)"),
    re.compile(r"(?:찾아|조회|검색)\s*(?:줘|해\s*줘|부탁)"),
)

_LOSS_ACTION_PATTERN = re.compile(r"(?:잃어버|분실(?:했|한|함)|두고|놓고)")
_NON_ITEM_TOKENS = {
    "어제",
    "오늘",
    "그제",
    "방금",
    "아까",
    "아침",
    "점심",
    "저녁",
    "새벽",
    "오전",
    "오후",
    "밤",
    "낮",
    "뭔가",
    "무언가",
    "물건",
    "이거",
    "그거",
}
_NON_ITEM_SUFFIXES = ("에서", "에게", "부터", "까지", "중에", "동안", "에")


def _message_content_text(content: Any) -> str:
    """문자열 또는 멀티모달 메시지에서 텍스트만 꺼낸다."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for block in content:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text", "")))
    return "\n".join(chunk for chunk in chunks if chunk)


def _latest_user_text(messages: list[Any]) -> str:
    """Memory 전체에서 현재 턴의 마지막 사용자 메시지를 찾는다."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_content_text(message.content)
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_content_text(message.get("content"))
    return ""


def is_explicit_report_request(text: str) -> bool:
    """사용자가 현재 메시지에서 신고서 도움을 분명히 요청했는지 판별한다."""
    normalized = re.sub(r"\s+", " ", text.strip())
    return any(pattern.search(normalized) for pattern in _REPORT_REQUEST_PATTERNS)


def is_explicit_search_request(text: str) -> bool:
    """물품을 명시해 바로 조회해 달라는 현재 턴의 표현을 판별한다."""
    if is_explicit_report_request(text):
        return False
    normalized = re.sub(r"\s+", " ", text.strip())
    if any(pattern.search(normalized) for pattern in _SEARCH_REQUEST_PATTERNS):
        return True

    # 구어체에서는 목적격 조사 없이 "핸드폰 잃어버렸어"라고 자주 말한다.
    # 분실 동사 직전 단어가 시간·장소 표현이 아니라면 물품명으로 간주한다.
    for action in _LOSS_ACTION_PATTERN.finditer(normalized):
        tokens = re.findall(r"[A-Za-z0-9가-힣]+", normalized[: action.start()])
        if not tokens:
            continue
        token = tokens[-1]
        if token.endswith(_NON_ITEM_SUFFIXES):
            continue
        candidate = re.sub(r"(?:을|를)$", "", token)
        if candidate and candidate not in _NON_ITEM_TOKENS:
            return True
    return False


def _search_tool_called_since_latest_user(messages: list[Any]) -> bool:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return False
        if (
            isinstance(message, ToolMessage)
            and message.name == "search_lost112_candidates"
        ):
            return True
    return False


def _tool_name(tool_definition: BaseTool | dict[str, Any]) -> str | None:
    if isinstance(tool_definition, BaseTool):
        return tool_definition.name
    if isinstance(tool_definition.get("name"), str):
        return tool_definition["name"]
    function = tool_definition.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


@wrap_model_call
def gate_report_tool_for_model(request: Any, handler: Any) -> Any:
    """명시적 신고서 요청이 없는 턴에는 모델에게 신고서 Tool을 숨긴다."""
    if is_explicit_report_request(_latest_user_text(request.messages)):
        return handler(request)

    tools = [
        tool_definition
        for tool_definition in request.tools
        if _tool_name(tool_definition) != "prepare_lost_report_draft"
    ]
    return handler(request.override(tools=tools))


@wrap_model_call
def force_explicit_search_tool_for_model(request: Any, handler: Any) -> Any:
    """물품이 명시된 조회 요청은 확인 질문 없이 검색 Tool을 선택한다."""
    user_text = _latest_user_text(request.messages)
    if (
        is_explicit_search_request(user_text)
        and not _search_tool_called_since_latest_user(request.messages)
    ):
        return handler(request.override(tool_choice="search_lost112_candidates"))
    return handler(request)


@wrap_tool_call
def block_unrequested_report_tool(request: Any, handler: Any) -> Any:
    """모델이 Tool 이름을 임의 생성해도 명시적 요청 없이는 실행하지 않는다."""
    if request.tool_call["name"] != "prepare_lost_report_draft":
        return handler(request)

    messages = request.state.get("messages", [])
    if is_explicit_report_request(_latest_user_text(messages)):
        return handler(request)

    return ToolMessage(
        content=(
            "현재 사용자 메시지에는 신고서 작성 요청이 없으므로 실행하지 않았습니다. "
            "습득물 조회 또는 사용자의 질문에만 답하세요."
        ),
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
    )


def _result_for_model(result: AgentResult) -> str:
    """LLM에는 사용자 안내에 필요한 필드만 전달해 컨텍스트를 제한한다."""
    payload = {
        "query": result.query.model_dump(mode="json", exclude_none=True),
        "source_counts": result.source_counts,
        "errors": result.errors,
        "search_scopes": [scope.model_dump(mode="json") for scope in result.search_scopes],
        "candidates": [
            {
                "score": candidate.score,
                "confidence": candidate.confidence,
                "location_scope": candidate.location_scope,
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
            tools=["search_lost112_candidates", "prepare_lost_report_draft"],
            on_failure="continue",
            initial_delay=0.5,
        ),
        ToolCallLimitMiddleware(
            tool_name="search_lost112_candidates", run_limit=1
        ),
        ToolCallLimitMiddleware(
            tool_name="prepare_lost_report_draft", run_limit=1
        ),
        force_explicit_search_tool_for_model,
        gate_report_tool_for_model,
        block_unrequested_report_tool,
        ModelCallLimitMiddleware(run_limit=4, exit_behavior="end"),
    ]


def _message_text(message: AIMessage | None) -> str:
    if message is None:
        return "답변을 생성하지 못했습니다."
    return _message_content_text(message.content) or "답변을 생성하지 못했습니다."


def _search_result_message(result: AgentResult) -> str:
    """모델이 후보의 URL이나 점수를 다시 쓰지 않도록 고정 안내문을 만든다."""
    if result.candidates:
        return f"습득물 후보 {len(result.candidates)}건을 찾았습니다."
    if result.source_counts:
        return "조회했지만 조건에 맞는 습득물 후보를 찾지 못했습니다."
    return "분실물 조회에 실패했습니다. 아래 오류 내용을 확인해주세요."


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
        self.report_tool = create_lost_report_tool()
        self.checkpointer = InMemorySaver()
        self.graph = create_agent(
            model=model,
            tools=[self.search_tool, self.report_tool],
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

        messages = state["messages"]
        current_turn_start = max(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage)
            ),
            default=-1,
        )

        final_message: AIMessage | None = None
        search_result: AgentResult | None = None
        report_draft: LostReportDraft | None = None
        for message in messages[current_turn_start + 1 :]:
            if isinstance(message, AIMessage):
                final_message = message
            elif (
                isinstance(message, ToolMessage)
                and message.name == self.search_tool.name
                and isinstance(message.artifact, AgentResult)
            ):
                search_result = message.artifact
            elif (
                isinstance(message, ToolMessage)
                and message.name == self.report_tool.name
                and isinstance(message.artifact, LostReportDraft)
            ):
                report_draft = message.artifact

        return JupJupChatResponse(
            # 후보 링크와 점수의 기준은 Tool artifact다. 모델의 자연어 요약은
            # URL 파라미터 등을 변형할 수 있으므로 검색 턴에는 노출하지 않는다.
            message=(
                _search_result_message(search_result)
                if search_result is not None
                else _message_text(final_message)
            ),
            search_result=search_result,
            report_draft=report_draft,
        )
