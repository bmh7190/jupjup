"""LangChain 실행 전후 및 모델·도구 호출 경계를 담당하는 미들웨어."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import (
    AgentState,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    after_agent,
    before_agent,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

from .intents import (
    has_search_item_context,
    is_explicit_search_request,
    is_explicit_similar_lost_report_request,
    is_report_workflow_turn,
    is_search_cancel_request,
    latest_user_text as _latest_user_text,
    report_context_from_messages,
    report_tool_called_since_latest_user,
    search_tool_called_since_latest_user,
    similar_reports_tool_called_since_latest_user,
)


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
    if is_report_workflow_turn(request.messages):
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
    if is_search_cancel_request(user_text):
        tools = [
            tool_definition
            for tool_definition in request.tools
            if _tool_name(tool_definition) != "search_lost112_candidates"
        ]
        return handler(request.override(tools=tools))
    if is_report_workflow_turn(request.messages):
        return handler(request)
    if not is_explicit_search_request(user_text):
        return handler(request)

    if not has_search_item_context(request.messages):
        tools = [
            tool_definition
            for tool_definition in request.tools
            if _tool_name(tool_definition) != "search_lost112_candidates"
        ]
        return handler(request.override(tools=tools))

    if not search_tool_called_since_latest_user(request.messages):
        return handler(request.override(tool_choice="search_lost112_candidates"))
    return handler(request)


@wrap_model_call
def route_similar_reports_for_model(request: Any, handler: Any) -> Any:
    """유사 신고 요청에서만 전용 Tool을 열고 다른 기능과 분리한다."""
    user_text = _latest_user_text(request.messages)
    is_similar_request = is_explicit_similar_lost_report_request(user_text)
    if not is_similar_request:
        tools = [
            tool_definition
            for tool_definition in request.tools
            if _tool_name(tool_definition) != "search_similar_lost_reports"
        ]
        return handler(request.override(tools=tools))

    tools = [
        tool_definition
        for tool_definition in request.tools
        if _tool_name(tool_definition)
        not in {"search_lost112_candidates", "prepare_lost_report_draft"}
    ]
    if not has_search_item_context(request.messages):
        return handler(request.override(tools=[]))
    if similar_reports_tool_called_since_latest_user(request.messages):
        return handler(request.override(tools=tools))
    return handler(
        request.override(
            tools=tools,
            tool_choice="search_similar_lost_reports",
        )
    )


@wrap_model_call
def route_report_workflow_for_model(request: Any, handler: Any) -> Any:
    """신고서 흐름에서는 검색 Tool을 숨기고 신고서 Tool을 확실히 실행한다."""
    if not is_report_workflow_turn(request.messages):
        return handler(request)

    tools = [
        tool_definition
        for tool_definition in request.tools
        if _tool_name(tool_definition)
        not in {"search_lost112_candidates", "search_similar_lost_reports"}
    ]
    if report_tool_called_since_latest_user(request.messages):
        return handler(request.override(tools=tools))
    return handler(
        request.override(
            tools=tools,
            tool_choice="prepare_lost_report_draft",
        )
    )


@wrap_tool_call
def block_unrequested_report_tool(request: Any, handler: Any) -> Any:
    """모델이 Tool 이름을 임의 생성해도 명시적 요청 없이는 실행하지 않는다."""
    if request.tool_call["name"] != "prepare_lost_report_draft":
        return handler(request)

    messages = request.state.get("messages", [])
    if is_report_workflow_turn(messages):
        return handler(request)

    return ToolMessage(
        content=(
            "현재 사용자 메시지에는 신고서 작성 요청이 없으므로 실행하지 않았습니다. "
            "습득물 조회 또는 사용자의 질문에만 답하세요."
        ),
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
    )


@wrap_tool_call
def block_search_without_item_name(request: Any, handler: Any) -> Any:
    """빈 물품명 검색을 Tool 검증 전에 막아 재시도와 API 호출을 피한다."""
    if request.tool_call["name"] not in {
        "search_lost112_candidates",
        "search_similar_lost_reports",
    }:
        return handler(request)

    item_name = request.tool_call.get("args", {}).get("item_name")
    if isinstance(item_name, str) and item_name.strip():
        return handler(request)

    return ToolMessage(
        content="검색할 물품명이 없습니다. 어떤 물품을 찾을지 먼저 물어보세요.",
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
    )


@wrap_tool_call
def merge_report_context(request: Any, handler: Any) -> Any:
    """신고서 수정 호출에서 모델이 생략한 이전 확인값을 보존한다."""
    if request.tool_call["name"] != "prepare_lost_report_draft":
        return handler(request)

    args = request.tool_call.get("args", {})
    if not isinstance(args, dict):
        return handler(request)
    previous = report_context_from_messages(request.state.get("messages", []))
    current_item_name = args.get("item_name")
    previous_item_name = previous.get("item_name")
    if (
        isinstance(current_item_name, str)
        and current_item_name.strip()
        and isinstance(previous_item_name, str)
        and previous_item_name.strip()
        and current_item_name.strip().casefold()
        != previous_item_name.strip().casefold()
    ):
        previous = {}
    merged = {**previous, **args}
    tool_call = {**request.tool_call, "args": merged}
    return handler(request.override(tool_call=tool_call))


@wrap_tool_call
def merge_similar_report_search_context(request: Any, handler: Any) -> Any:
    """유사 신고 후속 요청에 같은 대화에서 확인한 검색 조건을 채운다."""
    if request.tool_call["name"] != "search_similar_lost_reports":
        return handler(request)

    args = request.tool_call.get("args", {})
    if not isinstance(args, dict):
        return handler(request)
    allowed_fields = {
        "item_name",
        "category",
        "lost_date",
        "lost_time",
        "lost_place",
        "region",
        "color",
        "brand",
        "features",
    }
    previous = {
        key: value
        for key, value in report_context_from_messages(
            request.state.get("messages", [])
        ).items()
        if key in allowed_fields
    }
    current_item_name = args.get("item_name")
    previous_item_name = previous.get("item_name")
    if (
        isinstance(current_item_name, str)
        and current_item_name.strip()
        and isinstance(previous_item_name, str)
        and previous_item_name.strip()
        and current_item_name.strip().casefold()
        != previous_item_name.strip().casefold()
    ):
        previous = {}
    merged = {**previous, **args}
    tool_call = {**request.tool_call, "args": merged}
    return handler(request.override(tool_call=tool_call))



_FALSE_SUBMISSION_CLAIMS = (
    "신고를 접수했습니다",
    "신고가 접수되었습니다",
    "신고서를 제출했습니다",
    "민원이 접수되었습니다",
)


@before_agent(can_jump_to=["end"])
def validate_user_input(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """모델 호출 전에 빈 입력을 결정적으로 차단한다."""
    del runtime
    if _latest_user_text(state.get("messages", [])).strip():
        return None
    return {
        "messages": [AIMessage(content="메시지를 입력해주세요.")],
        "jump_to": "end",
    }


@after_agent
def validate_final_answer(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Agent 종료 뒤 실제로 하지 않은 신고 접수 완료 표현을 교정한다."""
    del runtime
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        return None
    content = messages[-1].content
    if not isinstance(content, str):
        return None
    if not any(claim in content for claim in _FALSE_SUBMISSION_CLAIMS):
        return None
    return {
        "messages": [
            AIMessage(
                content=(
                    "신고서는 자동 접수되지 않았습니다. 초안을 확인한 뒤 "
                    "경찰민원24에서 직접 접수해주세요."
                )
            )
        ]
    }


def build_middlewares() -> list[Any]:
    """입출력 개인정보 보호와 Agent/Tool 과호출 방지 정책."""
    return [
        validate_user_input,
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
            tools=[
                "search_lost112_candidates",
                "search_similar_lost_reports",
                "prepare_lost_report_draft",
            ],
            on_failure="continue",
            initial_delay=0.5,
        ),
        ToolCallLimitMiddleware(
            tool_name="search_lost112_candidates", run_limit=1
        ),
        ToolCallLimitMiddleware(
            tool_name="search_similar_lost_reports", run_limit=1
        ),
        ToolCallLimitMiddleware(
            tool_name="prepare_lost_report_draft", run_limit=1
        ),
        force_explicit_search_tool_for_model,
        route_similar_reports_for_model,
        route_report_workflow_for_model,
        gate_report_tool_for_model,
        block_unrequested_report_tool,
        merge_similar_report_search_context,
        block_search_without_item_name,
        merge_report_context,
        ModelCallLimitMiddleware(run_limit=4, exit_behavior="end"),
        validate_final_answer,
    ]
