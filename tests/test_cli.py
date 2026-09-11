from __future__ import annotations

import unittest
from io import StringIO
from unittest.mock import patch

from jupjup.agent import JupJupChatResponse
from jupjup.cli import print_chat_response, run_chat
from jupjup.models import (
    AgentResult,
    LostItemQuery,
    MatchCandidate,
    RecordSource,
    SearchRecord,
)


class StubAgent:
    def __init__(self, response: JupJupChatResponse) -> None:
        self.response = response

    def chat(self, text: str, *, thread_id: str) -> JupJupChatResponse:
        return self.response


class CliTest(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_json_output_excludes_telephone(self, stdout: StringIO) -> None:
        record = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="F1",
            item_name="지갑",
            telephone="02-1234-5678",
        )
        result = AgentResult(
            query=LostItemQuery(item_name="지갑", search_ready=True),
            candidates=[MatchCandidate(record=record, score=50, breakdown={})],
            similar_lost_reports=[],
            source_counts={"경찰청 습득물": 1},
        )

        print_chat_response(
            JupJupChatResponse(message="조회 완료", search_result=result), as_json=True
        )

        self.assertNotIn("telephone", stdout.getvalue())
        self.assertNotIn("02-1234-5678", stdout.getvalue())

    @patch("jupjup.cli.print_chat_response")
    def test_partial_api_success_has_zero_exit_status(self, _print) -> None:
        result = AgentResult(
            query=LostItemQuery(item_name="지갑", search_ready=True),
            candidates=[],
            similar_lost_reports=[],
            source_counts={"경찰청 습득물": 1},
            errors={"경찰청 분실물": "HTTP 에러"},
        )
        agent = StubAgent(JupJupChatResponse(message="조회 완료", search_result=result))

        self.assertEqual(run_chat(agent, "지갑", as_json=True), 0)  # type: ignore[arg-type]

    @patch("jupjup.cli.print_chat_response")
    def test_total_api_failure_has_nonzero_exit_status(self, _print) -> None:
        result = AgentResult(
            query=LostItemQuery(item_name="지갑", search_ready=True),
            candidates=[],
            similar_lost_reports=[],
            source_counts={},
            errors={"경찰청 분실물": "HTTP 에러"},
        )
        agent = StubAgent(JupJupChatResponse(message="조회 실패", search_result=result))

        self.assertEqual(run_chat(agent, "지갑", as_json=True), 1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
