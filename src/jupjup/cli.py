"""터미널에서 전체 흐름을 확인하는 실행 프로그램."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api_client import Lost112ApiClient
from .config import Settings
from .demo import run_demo
from .extractor import LostItemExtractor
from .models import AgentResult, LostItemQuery
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


def collect_query(extractor: LostItemExtractor, initial_text: str | None) -> LostItemQuery:
    messages: list[str] = []
    if initial_text:
        messages.append(f"사용자: {initial_text}")
    else:
        print("잃어버린 물건, 날짜와 장소를 자연스럽게 설명해주세요.")
        messages.append(f"사용자: {input('> ').strip()}")

    for _ in range(3):
        query = extractor.extract(messages)
        if query.search_ready and not query.missing_fields:
            return query
        if initial_text or not query.follow_up_question:
            return query
        print(f"\nAgent: {query.follow_up_question}")
        answer = input("> ").strip()
        if not answer:
            return query
        messages.append(f"Agent: {query.follow_up_question}")
        messages.append(f"사용자: {answer}")
    return extractor.extract(messages)


def print_result(result: AgentResult) -> None:
    print("\n[추출된 분실 정보]")
    print(result.query.model_dump_json(indent=2))

    print("\n[API 조회 건수]")
    for source, count in result.source_counts.items():
        print(f"- {source}: {count}건")
    for source, error in result.errors.items():
        print(f"- {source}: 조회 실패 ({error})")

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
        extractor = LostItemExtractor(
            model_name=settings.openai_model, api_key=settings.openai_api_key
        )
        query = collect_query(extractor, args.text)
        if not query.item_name:
            print("물품명을 확인하지 못해 검색을 시작할 수 없습니다.", file=sys.stderr)
            return 2

        api_client = Lost112ApiClient(
            settings.data_service_key,
            timeout_seconds=settings.timeout_seconds,
            page_size=settings.page_size,
            detail_limit=settings.detail_limit,
        )
        vision_matcher = None
        if args.vision:
            vision_matcher = VisionMatcher(
                model_name=settings.openai_model, api_key=settings.openai_api_key
            )
        service = JupJupAgentService(api_client, vision_matcher=vision_matcher)
        result = service.run(query)
    except (ValueError, RuntimeError) as exc:
        print(f"실행 오류: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print_result(result)
    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

