"""JupJup 대화 Agent 구성과 응답 조립."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from ..application.service import JupJupAgentService
from ..domain.models import AgentResult, LostReportDraft
from .intents import message_content_text as _message_content_text
from .middleware import build_middlewares
from .tools import (
    create_lost112_search_tool,
    create_lost_report_tool,
    create_similar_lost_reports_tool,
)
SYSTEM_PROMPT = """당신은 분실물 찾기를 돕는 '줍줍이'입니다.
사용자와 한국어로 짧고 명확하게 대화하세요.

다음 원칙을 지키세요.
- 먼저 사용자의 의도를 습득물 조회, 유사 분실 신고 조회, 분실신고 작성 도움으로 구분하세요.
- 사용자의 설명에서 물품명, 분실일, 장소, 지역, 색상, 크기, 브랜드, 수량, 특징을 파악하세요.
- region은 광역 시도명으로 통일하세요. 예: 강남역·광진구는 서울, 우도는 제주입니다.
- item_name에는 '지갑'처럼 짧은 일반 물품명을 넣고 '샤넬' 같은 브랜드와 색상은 별도 인자로 전달하세요.
- 사용자가 분실신고 작성, 문장 정리, 누락 확인을 명시적으로 시작한 경우에만 prepare_lost_report_draft Tool을 호출하세요.
- 같은 대화에서 신고서에 필요한 정보를 답하거나 초안 수정을 요청하면, 이전에 확인한 신고 정보와 합쳐 prepare_lost_report_draft Tool을 다시 호출하세요.
- 사용자가 신고서 작성을 취소하거나 습득물 검색으로 전환하거나 대화를 마무리하면 신고서 Tool을 더 호출하지 마세요.
- 단순히 물건을 잃어버렸다고 설명하거나 습득물 조회를 요청한 경우에는 신고서 Tool을 호출하지 마세요.
- 신고서 작성만 요청한 경우에는 검색 Tool을 호출하지 마세요.
- 신고서 작성 요청에서는 물품명, 분실 날짜, 구체적인 장소가 없으면 초안의 next_question으로 먼저 보완하세요. 필수 정보가 모이기 전에는 초안을 준비했다고 말하지 마세요.
- 시간, 지역, 색상, 크기, 브랜드, 수량, 특징, 분실 경위는 선택 정보입니다. 필수 정보처럼 답변을 요구하지 말고 완성된 초안의 개선 제안으로만 안내하세요.
- prepare_lost_report_draft 결과가 준비되면 복사용 문장, 누락 항목, 개선 제안, 주의사항을 안내하세요.
- 신고서 결과의 official_report_url과 official_guide_url을 답변에서 생략하지 마세요.
- 이 서비스는 신고를 자동 제출하지 않습니다. 제출됐다고 말하지 말고 경찰민원24 공식 링크를 안내하세요.
- 도난은 분실물 신고와 구분하고, 자동차번호판은 방문 신고가 필요하다는 Tool 결과를 따르세요.
- 조회 요청에서 물품명을 알 수 없으면 검색하지 말고 먼저 물어보세요.
- 조회 요청에서 물품명을 알 수 있으면 확인이나 동의를 다시 묻지 말고 같은 턴에 search_lost112_candidates Tool로 실제 데이터를 조회하세요.
- 일반 습득물 조회에는 search_lost112_candidates만 사용하세요. 이 Tool은 경찰청 습득물과 포털기관 습득물만 조회합니다.
- 사용자가 다른 사람의 유사하거나 비슷한 분실 신고를 확인해 달라고 명시한 경우에만 search_similar_lost_reports를 사용하세요.
- 유사 분실 신고 요청에 물품명이 생략되면 같은 대화에서 앞서 확인한 물품 조건을 사용하세요. 이전 조건도 없으면 물품명을 먼저 물어보세요.
- 사용자가 제공하지 않은 조건은 추측해서 Tool 인자에 넣지 마세요.
- 조회하지 않은 결과를 찾았다고 말하지 마세요.
- 습득물 후보와 다른 사람이 등록한 유사 분실 신고를 구분하세요.
- 후보 점수·신뢰도·일치 근거는 내부 정렬에만 사용하고 사용자에게 노출하지 마세요.
- 후보에는 물품명, 색상·분류, 습득 날짜·시간·장소, 보관 장소·상태, 기관명, 사진과 상세 URL을 사용하세요.
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


def _similar_lost_reports_message(result: AgentResult) -> str:
    """명시적으로 요청한 유사 분실 신고를 짧은 채팅 답변으로 만든다."""
    if not result.similar_lost_reports:
        if result.errors and not result.source_counts:
            return "유사한 기존 분실 신고를 조회하지 못했습니다. 잠시 후 다시 시도해주세요."
        return "같은 조건으로 등록된 유사한 기존 분실 신고를 찾지 못했습니다."

    lines = [
        f"유사한 기존 분실 신고 {len(result.similar_lost_reports)}건을 찾았습니다."
    ]
    for record in result.similar_lost_reports:
        lines.append(
            f"- {record.item_name or '이름 없음'} / "
            f"{record.event_date or '-'} / {record.event_place or '-'}"
        )
        if record.detail_url:
            lines.append(f"  상세: {record.detail_url}")
    return "\n".join(lines)


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
        self.similar_reports_tool = create_similar_lost_reports_tool(service)
        self.report_tool = create_lost_report_tool()
        self.checkpointer = InMemorySaver()
        self.graph = create_agent(
            model=model,
            tools=[self.search_tool, self.similar_reports_tool, self.report_tool],
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
        similar_reports_result: AgentResult | None = None
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
                and message.name == self.similar_reports_tool.name
                and isinstance(message.artifact, AgentResult)
            ):
                similar_reports_result = message.artifact
            elif (
                isinstance(message, ToolMessage)
                and message.name == self.report_tool.name
                and isinstance(message.artifact, LostReportDraft)
            ):
                report_draft = message.artifact

        needs_report_details = bool(
            report_draft is not None and report_draft.missing_essential_fields
        )
        if search_result is not None:
            response_message = _search_result_message(search_result)
        elif similar_reports_result is not None:
            response_message = _similar_lost_reports_message(similar_reports_result)
        elif needs_report_details and report_draft is not None:
            response_message = report_draft.next_question or (
                "신고서 초안을 만들려면 필요한 분실 정보를 알려주세요."
            )
        else:
            response_message = _message_text(final_message)

        return JupJupChatResponse(
            # 후보 링크와 점수의 기준은 Tool artifact다. 모델의 자연어 요약은
            # URL 파라미터 등을 변형할 수 있으므로 검색 턴에는 노출하지 않는다.
            message=response_message,
            search_result=search_result,
            # Tool artifact는 Memory 안에 남아 다음 답변과 합쳐진다. 필수 정보가
            # 부족한 동안에는 질문만 노출하고, 준비된 뒤에 초안 카드를 반환한다.
            report_draft=None if needs_report_details else report_draft,
        )
