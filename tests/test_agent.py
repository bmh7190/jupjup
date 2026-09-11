from __future__ import annotations

import unittest
from datetime import date
from typing import Any

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from jupjup.agent import (
    JupJupChatAgent,
    _search_result_message,
    build_middlewares,
    create_lost112_search_tool,
    create_lost_report_tool,
    has_search_item_context,
    is_explicit_report_request,
    is_explicit_search_request,
    is_report_workflow_turn,
)
from jupjup.conversation import report_context_from_messages
from jupjup.models import AgentResult, LostItemQuery
from jupjup.report import build_lost_report_draft


class StubService:
    def __init__(self) -> None:
        self.received: LostItemQuery | None = None

    def run(self, query: LostItemQuery, *, candidate_limit: int = 5) -> AgentResult:
        self.received = query
        return AgentResult(
            query=query,
            candidates=[],
            similar_lost_reports=[],
            source_counts={"경찰청 습득물": 0},
        )


class ScriptedToolModel(BaseChatModel):
    """실제 API 호출 없이 Agent의 대화와 Tool loop를 검증한다."""

    _step: int = PrivateAttr(default=0)
    _received_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def bind_tools(
        self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any
    ) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._received_messages.append(messages)
        self._step += 1
        if self._step == 1:
            message = AIMessage(content="어떤 물건을 잃어버리셨나요?")
        elif self._step == 2:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_lost112_candidates",
                        "args": {"item_name": "카드지갑", "lost_place": "강남역"},
                        "id": "search-call-1",
                    }
                ],
            )
        else:
            message = AIMessage(
                content=(
                    "상세: https://minwon24.police.go.kr/example?"
                    "pkupCmdId=모델이잘못바꾼값"
                )
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


class EchoModel(BaseChatModel):
    """Middleware가 모델 앞뒤에서 텍스트를 가리는지 확인하는 모델."""

    @property
    def _llm_type(self) -> str:
        return "echo-model"

    def bind_tools(
        self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any
    ) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=messages[-1].content))]
        )


class ToolChoiceAwareModel(BaseChatModel):
    """강제된 Tool 선택을 따르고 이후에는 최종 답변을 반환한다."""

    _tool_choice: str | None = PrivateAttr(default=None)
    _tool_choices: list[str | None] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "tool-choice-aware-model"

    def bind_tools(
        self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any
    ) -> BaseChatModel:
        self._tool_choice = tool_choice
        self._tool_choices.append(tool_choice)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._tool_choice == "search_lost112_candidates":
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_lost112_candidates",
                        "args": {
                            "item_name": "카드지갑",
                            "lost_place": "강남역",
                        },
                        "id": "forced-search-call-1",
                    }
                ],
            )
        else:
            message = AIMessage(content="검색을 완료했습니다.")
        return ChatResult(generations=[ChatGeneration(message=message)])


class ScriptedReportModel(BaseChatModel):
    """신고서 Tool의 Agent 연결을 외부 모델 호출 없이 검증한다."""

    _step: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-report-model"

    def bind_tools(
        self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any
    ) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._step += 1
        if self._step == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "prepare_lost_report_draft",
                        "args": {
                            "item_name": "카드지갑",
                            "lost_date": "2026-09-10",
                            "lost_place": "강남역",
                            "size": "가로 11cm",
                            "quantity": 1,
                        },
                        "id": "report-call-1",
                    }
                ],
            )
        else:
            message = AIMessage(content="신고서 초안을 준비했습니다.")
        return ChatResult(generations=[ChatGeneration(message=message)])


class ReportConversationModel(BaseChatModel):
    """현재 턴에 공개된 Tool만 사용해 신고서 연속 수정을 재현한다."""

    _tool_names: set[str] = PrivateAttr(default_factory=set)
    _tool_choices: list[str | None] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "report-conversation-model"

    def bind_tools(
        self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any
    ) -> BaseChatModel:
        self._tool_names = {
            tool.name if hasattr(tool, "name") else tool.get("name")
            for tool in tools
        }
        self._tool_choices.append(tool_choice)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(messages[-1], ToolMessage):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="초안을 갱신했습니다."))]
            )

        user_text = next(
            message.content
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        if "prepare_lost_report_draft" not in self._tool_names:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="일반 답변입니다."))]
            )

        args: dict[str, Any]
        if "휴대폰" in user_text:
            args = {"item_name": "휴대폰"}
        elif "시간" in user_text and "빼" in user_text:
            args = {"lost_time": None}
        elif "12cm" in user_text:
            args = {"size": "가로 12cm"}
        elif "작성" in user_text:
            args = {"incident_type": "분실"}
        elif "파란색" in user_text:
            args = {
                "item_name": "지갑",
                "lost_date": "2026-09-10",
                "lost_place": "강남역 2번 출구",
                "color": "파란색",
                "size": "가로 11cm",
            }
        else:
            args = {
                "item_name": "지갑",
                "lost_date": "2026-09-10",
                "lost_time": "18:00",
                "lost_place": "강남역 2번 출구",
            }
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "prepare_lost_report_draft",
                                "args": args,
                                "id": f"report-conversation-{len(messages)}",
                            }
                        ],
                    )
                )
            ]
        )


class BlankSearchModel(BaseChatModel):
    """숨긴 Tool을 임의 호출해도 빈 검색이 실행되지 않는지 확인한다."""

    @property
    def _llm_type(self) -> str:
        return "blank-search-model"

    def bind_tools(
        self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any
    ) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(messages[-1], ToolMessage):
            message = AIMessage(content="어떤 물품을 찾을까요?")
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_lost112_candidates",
                        "args": {"item_name": ""},
                        "id": "blank-search",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


class AgentConfigurationTest(unittest.TestCase):
    def test_agent_returns_report_draft_artifact(self) -> None:
        agent = JupJupChatAgent(StubService(), model=ScriptedReportModel())  # type: ignore[arg-type]

        response = agent.chat("분실신고서를 써줘", thread_id="report-test")

        self.assertEqual(response.message, "신고서 초안을 준비했습니다.")
        self.assertIsNotNone(response.report_draft)
        assert response.report_draft is not None
        self.assertEqual(response.report_draft.item_name, "카드지갑")
        self.assertEqual(response.report_draft.size, "가로 11cm")
        self.assertEqual(response.report_draft.quantity, 1)
        self.assertIn("cvlcptAply.do", response.report_draft.official_report_url)
        self.assertFalse(response.report_draft.auto_submitted)

        following_response = agent.chat("고마워", thread_id="report-test")

        self.assertIsNone(following_response.report_draft)

    def test_report_tool_is_blocked_without_explicit_user_request(self) -> None:
        agent = JupJupChatAgent(StubService(), model=ScriptedReportModel())  # type: ignore[arg-type]

        response = agent.chat("강남역에서 카드지갑을 잃어버렸어", thread_id="no-report-test")

        self.assertIsNone(response.report_draft)

    def test_report_request_requires_an_explicit_action(self) -> None:
        self.assertFalse(is_explicit_report_request("강남역에서 지갑을 잃어버렸어"))
        self.assertFalse(is_explicit_report_request("분실 신고는 어디에서 해?"))
        self.assertTrue(is_explicit_report_request("분실신고서 초안을 써줘"))
        self.assertTrue(is_explicit_report_request("빠진 항목이 있는지 확인해줘"))
        self.assertFalse(
            is_explicit_report_request("신고서는 작성하지 말고 습득물만 찾아줘")
        )

    def test_report_workflow_keeps_only_report_follow_up_turns_active(self) -> None:
        start = HumanMessage(content="분실신고서 작성 도와줘")
        question = AIMessage(content="신고서에 넣을 물품, 날짜, 장소를 알려주세요.")

        self.assertTrue(
            is_report_workflow_turn(
                [
                    start,
                    question,
                    HumanMessage(
                        content="어제 강남역에서 지갑을 잃어버렸어"
                    ),
                ]
            )
        )
        self.assertTrue(
            is_report_workflow_turn(
                [start, question, HumanMessage(content="고마워. 파란색으로 수정해줘")]
            )
        )
        self.assertFalse(
            is_report_workflow_turn(
                [start, question, HumanMessage(content="오늘 날씨 어때?")]
            )
        )
        self.assertFalse(
            is_report_workflow_turn(
                [start, question, HumanMessage(content="그만하고 지갑을 찾아줘")]
            )
        )
        self.assertFalse(
            is_report_workflow_turn(
                [
                    start,
                    question,
                    HumanMessage(content="신고서 작성하지 마. 취소할게"),
                    AIMessage(content="취소했습니다."),
                    HumanMessage(content="파란색이야"),
                ]
            )
        )
        self.assertFalse(
            is_report_workflow_turn([HumanMessage(content="파란색으로 수정해줘")])
        )

    def test_report_follow_up_updates_draft_in_same_thread_only(self) -> None:
        model = ReportConversationModel()
        agent = JupJupChatAgent(StubService(), model=model)  # type: ignore[arg-type]

        missing = agent.chat("분실신고서 작성 도와줘", thread_id="report-flow")
        completed = agent.chat(
            "지갑이고 2026년 9월 10일에 강남역 2번 출구에서 잃어버렸어",
            thread_id="report-flow",
        )
        edited = agent.chat(
            "검정이 아니라 파란색이고 가로 11cm야. 수정해줘",
            thread_id="report-flow",
        )
        unrelated = agent.chat("오늘 날씨 어때?", thread_id="report-flow")
        isolated = agent.chat("파란색으로 수정해줘", thread_id="other-thread")

        self.assertIsNotNone(missing.report_draft)
        self.assertFalse(missing.report_draft.ready_for_user_review)  # type: ignore[union-attr]
        self.assertTrue(completed.report_draft.ready_for_user_review)  # type: ignore[union-attr]
        self.assertEqual(edited.report_draft.color, "파란색")  # type: ignore[union-attr]
        self.assertEqual(edited.report_draft.size, "가로 11cm")  # type: ignore[union-attr]
        self.assertEqual(edited.report_draft.lost_time, "18:00")  # type: ignore[union-attr]
        self.assertNotIn("search_lost112_candidates", model._tool_choices)
        self.assertIsNone(unrelated.report_draft)
        self.assertIsNone(isolated.report_draft)

        cleared = agent.chat("분실 시간은 빼줘", thread_id="report-flow")
        after_clear = agent.chat("크기는 가로 12cm로 수정해줘", thread_id="report-flow")
        new_incident = agent.chat(
            "이번엔 휴대폰 신고서 작성 도와줘", thread_id="report-flow"
        )

        self.assertIsNone(cleared.report_draft.lost_time)  # type: ignore[union-attr]
        self.assertIsNone(after_clear.report_draft.lost_time)  # type: ignore[union-attr]
        self.assertEqual(after_clear.report_draft.size, "가로 12cm")  # type: ignore[union-attr]
        self.assertEqual(new_incident.report_draft.item_name, "휴대폰")  # type: ignore[union-attr]
        self.assertIsNone(new_incident.report_draft.lost_date)  # type: ignore[union-attr]
        self.assertIsNone(new_incident.report_draft.lost_place)  # type: ignore[union-attr]

    def test_search_request_requires_an_explicit_item_or_action(self) -> None:
        self.assertTrue(is_explicit_search_request("강남역에서 카드지갑을 잃어버렸어"))
        self.assertTrue(is_explicit_search_request("어제 분당구에서 핸드폰 잃어버렸어"))
        self.assertTrue(is_explicit_search_request("카페에 가방 두고 왔어"))
        self.assertTrue(is_explicit_search_request("검은 우산 조회해줘"))
        self.assertFalse(is_explicit_search_request("강남역에서 잃어버렸어요"))
        self.assertFalse(is_explicit_search_request("어제 저녁에 잃어버렸어요"))
        self.assertFalse(is_explicit_search_request("뭔가 잃어버렸어요"))
        self.assertFalse(is_explicit_search_request("분실신고서를 작성해줘"))

    def test_search_item_context_uses_sentence_structure_and_thread_history(self) -> None:
        self.assertFalse(
            has_search_item_context([HumanMessage(content="기능 검증용 대화야. 찾아줘")])
        )
        self.assertTrue(
            has_search_item_context(
                [HumanMessage(content="파란 텀블러야. 강남역에서 잃어버렸어")]
            )
        )
        self.assertTrue(
            has_search_item_context(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_lost112_candidates",
                                "args": {"item_name": "키링"},
                                "id": "previous-search",
                            }
                        ],
                    ),
                    ToolMessage(
                        content="조회 완료",
                        tool_call_id="previous-search",
                        name="search_lost112_candidates",
                        artifact=AgentResult(
                            query=LostItemQuery(item_name="키링", search_ready=True),
                            candidates=[],
                            similar_lost_reports=[],
                            source_counts={"경찰청 습득물": 0},
                        ).model_dump(mode="json"),
                    ),
                    HumanMessage(content="같은 조건으로 다시 찾아줘"),
                ]
            )
        )

    def test_bare_search_request_asks_before_forcing_tool(self) -> None:
        service = StubService()
        model = ToolChoiceAwareModel()
        agent = JupJupChatAgent(service, model=model)  # type: ignore[arg-type]

        response = agent.chat("찾아줘", thread_id="bare-search")

        self.assertEqual(response.message, "검색을 완료했습니다.")
        self.assertIsNone(response.search_result)
        self.assertIsNone(service.received)
        self.assertNotIn("search_lost112_candidates", model._tool_choices)

    def test_invented_blank_search_call_is_blocked_before_service(self) -> None:
        service = StubService()
        agent = JupJupChatAgent(service, model=BlankSearchModel())  # type: ignore[arg-type]

        response = agent.chat("찾아줘", thread_id="blocked-blank-search")

        self.assertEqual(response.message, "어떤 물품을 찾을까요?")
        self.assertIsNone(response.search_result)
        self.assertIsNone(service.received)

    def test_report_context_uses_only_successful_tool_call_arguments(self) -> None:
        draft = build_lost_report_draft(
            item_name="지갑",
            lost_date=date(2026, 9, 10),
            lost_time="18:00",
            lost_place="강남역",
        )
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "prepare_lost_report_draft",
                        "args": {"item_name": "지갑", "lost_time": "18:00"},
                        "id": "successful-report",
                    }
                ],
            ),
            ToolMessage(
                content="완료",
                tool_call_id="successful-report",
                name="prepare_lost_report_draft",
                artifact=draft.model_dump(mode="json"),
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "prepare_lost_report_draft",
                        "args": {"lost_time": None, "color": "추측값"},
                        "id": "failed-report",
                    }
                ],
            ),
            ToolMessage(
                content="검증 실패",
                tool_call_id="failed-report",
                name="prepare_lost_report_draft",
            ),
        ]

        context = report_context_from_messages(messages)

        self.assertEqual(context["lost_time"], "18:00")
        self.assertNotIn("color", context)

    def test_explicit_item_loss_forces_search_without_confirmation(self) -> None:
        service = StubService()
        model = ToolChoiceAwareModel()
        agent = JupJupChatAgent(service, model=model)  # type: ignore[arg-type]

        response = agent.chat(
            "어제 분당구에서 핸드폰 잃어버렸어",
            thread_id="forced-search-test",
        )

        self.assertEqual(
            response.message,
            "조회했지만 조건에 맞는 습득물 후보를 찾지 못했습니다.",
        )
        self.assertIsNotNone(response.search_result)
        self.assertIsNotNone(service.received)
        self.assertIn("search_lost112_candidates", model._tool_choices)

    def test_report_tool_builds_reviewable_draft_without_submitting(self) -> None:
        report_tool = create_lost_report_tool()

        result = report_tool.invoke(
            {
                "item_name": "카드지갑",
                "lost_date": "2026-09-10",
                "lost_time": "18:30경",
                "lost_place": "강남역 2호선 승강장",
                "color": "검정",
                "features": ["흰색 스티치"],
            }
        )

        self.assertEqual(report_tool.name, "prepare_lost_report_draft")
        self.assertIn("ready_for_user_review", result)
        self.assertIn('"auto_submitted":false', result)

    def test_custom_tool_builds_query_and_calls_service(self) -> None:
        service = StubService()
        search_tool = create_lost112_search_tool(service)  # type: ignore[arg-type]

        content = search_tool.invoke(
            {
                "item_name": "카드지갑",
                "lost_date": "2026-09-10",
                "lost_place": "강남역",
                "color": "검정",
                "features": ["흰색 스티치"],
            }
        )

        self.assertEqual(search_tool.name, "search_lost112_candidates")
        self.assertIsNotNone(service.received)
        assert service.received is not None
        self.assertEqual(service.received.item_name, "카드지갑")
        self.assertEqual(service.received.lost_date, date(2026, 9, 10))
        self.assertIn("source_counts", content)

    def test_search_summary_is_generated_from_structured_result(self) -> None:
        result = AgentResult(
            query=LostItemQuery(item_name="지갑", search_ready=True),
            candidates=[],
            similar_lost_reports=[],
            source_counts={},
            errors={"경찰청 습득물": "HTTP 오류"},
        )

        self.assertEqual(
            _search_result_message(result),
            "분실물 조회에 실패했습니다. 아래 오류 내용을 확인해주세요.",
        )

    def test_agent_has_pii_retry_and_call_limit_middlewares(self) -> None:
        middlewares = build_middlewares()

        self.assertEqual(sum(isinstance(item, PIIMiddleware) for item in middlewares), 4)
        self.assertTrue(any(isinstance(item, ToolRetryMiddleware) for item in middlewares))
        self.assertTrue(any(isinstance(item, ToolCallLimitMiddleware) for item in middlewares))
        self.assertTrue(any(isinstance(item, ModelCallLimitMiddleware) for item in middlewares))

    def test_agent_calls_tool_and_remembers_previous_turn(self) -> None:
        service = StubService()
        model = ScriptedToolModel()
        agent = JupJupChatAgent(service, model=model)  # type: ignore[arg-type]

        first = agent.chat("강남역에서 잃어버렸어요", thread_id="memory-test")
        second = agent.chat("검은색 카드지갑이에요", thread_id="memory-test")

        self.assertEqual(first.message, "어떤 물건을 잃어버리셨나요?")
        self.assertEqual(
            second.message,
            "조회했지만 조건에 맞는 습득물 후보를 찾지 못했습니다.",
        )
        self.assertNotIn("pkupCmdId", second.message)
        self.assertIsNotNone(second.search_result)
        self.assertEqual(service.received.item_name, "카드지갑")  # type: ignore[union-attr]
        second_turn_inputs = model._received_messages[1]
        human_messages = [
            message.content
            for message in second_turn_inputs
            if isinstance(message, HumanMessage)
        ]
        self.assertEqual(
            human_messages,
            ["강남역에서 잃어버렸어요", "검은색 카드지갑이에요"],
        )

    def test_pii_middleware_redacts_phone_and_email_in_korean_sentence(self) -> None:
        agent = JupJupChatAgent(StubService(), model=EchoModel())  # type: ignore[arg-type]

        response = agent.chat(
            "연락처는 010-1234-5678이고 이메일은 test@example.com입니다.",
            thread_id="pii-test",
        )

        self.assertNotIn("010-1234-5678", response.message)
        self.assertNotIn("test@example.com", response.message)
        self.assertIn("[REDACTED_PHONE_NUMBER]", response.message)
        self.assertIn("[REDACTED_EMAIL]", response.message)


if __name__ == "__main__":
    unittest.main()
