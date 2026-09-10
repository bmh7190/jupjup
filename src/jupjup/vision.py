"""선택 기능: 습득물 사진이 사용자 설명과 맞는지 멀티모달 모델로 판정한다."""

from __future__ import annotations

from .models import LostItemQuery, SearchRecord, VisionAssessment


class VisionMatcher:
    def __init__(self, *, model_name: str, api_key: str) -> None:
        from langchain_openai import ChatOpenAI

        self._model = ChatOpenAI(
            model=model_name, api_key=api_key, temperature=0
        ).with_structured_output(VisionAssessment, method="json_schema")

    def score(self, query: LostItemQuery, record: SearchRecord) -> VisionAssessment | None:
        if not record.image_url:
            return None
        from langchain_core.messages import HumanMessage

        description = query.model_dump_json(exclude={"missing_fields", "follow_up_question"})
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "다음 분실물 설명과 습득물 사진이 같은 물건일 가능성을 0~1로 평가하세요. "
                        "사진으로 확인할 수 없는 날짜·장소는 판단 근거에서 제외하세요.\n"
                        f"분실물 설명: {description}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": record.image_url}},
            ]
        )
        result = self._model.invoke([message])
        if isinstance(result, VisionAssessment):
            return result
        return VisionAssessment.model_validate(result)

