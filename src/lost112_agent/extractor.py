"""LangChain Structured Output으로 대화에서 검색 조건을 추출한다."""

from __future__ import annotations

from datetime import date
from typing import Iterable

from .models import LostItemQuery
from .privacy import mask_pii


class LostItemExtractor:
    def __init__(self, *, model_name: str, api_key: str) -> None:
        if not api_key:
            raise ValueError("자연어 추출에는 OPENAI_API_KEY가 필요합니다.")

        # 선택 의존성은 실제 LLM 기능을 사용할 때만 불러와 테스트를 가볍게 유지한다.
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
당신은 한국 경찰 유실물 검색을 돕는 정보 추출기입니다.
대화에서 사용자가 잃어버린 물건의 정보를 추출하세요.

규칙:
- 오늘 날짜는 {today}입니다. '어제', '지난주 금요일' 같은 표현을 실제 날짜로 변환하세요.
- 사용자가 말하지 않은 값은 추측하지 말고 null 또는 빈 목록으로 남기세요.
- item_name은 API 검색에 적합한 짧은 보통명사로 작성하세요. 예: '검은색 카드지갑' -> '지갑'.
- category는 알 수 있을 때만 작성하세요.
- 식별에 유용한 색상, 브랜드, 모양, 흠집, 내용물은 각각 분리하세요.
- 이름, 연락처, 주민등록번호, 카드번호는 결과에 포함하지 마세요.
- 검색 최소 조건은 item_name입니다. item_name이 있으면 search_ready=true입니다.
- 날짜, 장소, 색상, 특징 중 빠진 값은 missing_fields에 기록하세요.
- 추가 질문은 한 번에 가장 중요한 정보 1~2개만 자연스럽게 물으세요.
                    """.strip(),
                ),
                ("human", "지금까지의 대화:\n{conversation}"),
            ]
        )
        llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
        self._chain = prompt | llm.with_structured_output(LostItemQuery, method="json_schema")

    def extract(self, messages: Iterable[str]) -> LostItemQuery:
        conversation = "\n".join(
            f"{index + 1}. {mask_pii(message)}" for index, message in enumerate(messages)
        )
        result = self._chain.invoke({"today": date.today().isoformat(), "conversation": conversation})
        if not isinstance(result, LostItemQuery):
            result = LostItemQuery.model_validate(result)
        return result

