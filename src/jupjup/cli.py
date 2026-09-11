"""터미널에서 전체 흐름을 확인하는 실행 프로그램."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from langchain_core.exceptions import ModelAuthenticationError

from .agent import JupJupChatAgent, JupJupChatResponse
from .api_client import Lost112ApiClient
from .config import Settings
from .demo import run_demo
from .models import AgentResult
from .service import JupJupAgentService
from .vision import VisionMatcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JupJup(줍줍) 분실물 자동 매칭 Agent")
    parser.add_argument("--text", help="대화형 입력 대신 한 문장으로 실행")
    parser.add_argument("--env", type=Path, help="사용할 .env 파일 경로")
    parser.add_argument("--vision", action="store_true", help="상위 후보 사진을 LLM으로 추가 판정")
    parser.add_argument("--demo", action="store_true", help="외부 API 없이 예시 데이터로 실행")
    parser.add_argument("--json", action="store_true", help="최종 결과를 JSON으로 출력")
    return parser


def print_result(result: AgentResult) -> None:
    print("\n[추출된 분실 정보]")
    print(result.query.model_dump_json(indent=2))

    print("\n[API 조회 건수]")
    for source, count in result.source_counts.items():
        print(f"- {source}: {count}건")
    for source, error in result.errors.items():
        print(f"- {source}: 조회 실패 ({error})")

    if result.search_scopes:
        print("\n[실제 조회 범위]")
        for scope in result.search_scopes:
            status = "조회 완료" if scope.complete else "일부 조회/오류"
            print(f"- {scope.source.label}: {scope.start_date}~{scope.end_date}, "
                  f"{scope.pages_completed}페이지·{scope.retrieved_count}건 확인 ({status})")

    print("\n[추천 습득물 후보]")
    if not result.candidates:
        print("조건에 맞는 후보를 찾지 못했습니다.")
    for index, candidate in enumerate(result.candidates, start=1):
        record = candidate.record
        print(f"\n{index}. {record.item_name or '이름 없음'} — {candidate.score}점")
        print(f"   출처: {record.source.label}")
        print(f"   날짜/장소: {record.event_date or '-'} / {record.event_place or '-'}")
        print(f"   보관장소: {record.custody_place or '-'}")
        print(f"   근거: {', '.join(candidate.reasons)}")
        print(f"   사진: {record.image_url or '등록된 사진 없음'}")
        print(f"   상세: {record.detail_url or '-'}")

    if result.similar_lost_reports:
        print("\n[유사한 기존 분실 신고]")
        for record in result.similar_lost_reports:
            print(f"- {record.item_name} / {record.event_date or '-'} / {record.event_place or '-'}")


def print_chat_response(response: JupJupChatResponse, *, as_json: bool) -> None:
    if as_json:
        print(response.model_dump_json(indent=2))
        return
    print(f"\n줍줍이: {response.message}")
    if response.search_result:
        print_result(response.search_result)


def run_chat(agent: JupJupChatAgent, initial_text: str | None, *, as_json: bool) -> int:
    """같은 thread_id를 사용해 여러 입력 사이의 대화를 기억한다."""
    thread_id = str(uuid4())
    if initial_text:
        response = agent.chat(initial_text, thread_id=thread_id)
        print_chat_response(response, as_json=as_json)
        return 0 if not response.search_result or not response.search_result.errors else 1

    print("잃어버린 물건을 설명해주세요. 종료하려면 '종료'를 입력하세요.")
    while True:
        user_text = input("> ").strip()
        if not user_text or user_text.lower() in {"종료", "quit", "exit"}:
            return 0
        response = agent.chat(user_text, thread_id=thread_id)
        print_chat_response(response, as_json=as_json)


def main() -> int:
    args = build_parser().parse_args()
    if args.demo:
        result = run_demo()
        print(result.model_dump_json(indent=2) if args.json else "", end="")
        if not args.json:
            print_result(result)
        return 0

    try:
        settings = Settings.from_env(args.env)
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 없습니다. .env.example을 참고하세요.")

        api_client = Lost112ApiClient(
            settings.data_service_key,
            **settings.api_client_options,
        )
        vision_matcher = None
        if args.vision:
            vision_matcher = VisionMatcher(
                model_name=settings.openai_model, api_key=settings.openai_api_key
            )
        service = JupJupAgentService(api_client, vision_matcher=vision_matcher)
        agent = JupJupChatAgent(
            service,
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
        )
        return run_chat(agent, args.text, as_json=args.json)
    except ModelAuthenticationError:
        print(
            "OpenAI 인증에 실패했습니다. 프로젝트 .env의 OPENAI_API_KEY를 "
            "새로 발급한 키로 확인하세요.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"실행 오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
