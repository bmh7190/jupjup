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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from jupjup.agent import JupJupChatAgent, build_middlewares, create_lost112_search_tool
from jupjup.models import AgentResult, LostItemQuery


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
            message = AIMessage(content="검색을 완료했습니다.")
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


class AgentConfigurationTest(unittest.TestCase):
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
        self.assertEqual(second.message, "검색을 완료했습니다.")
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
