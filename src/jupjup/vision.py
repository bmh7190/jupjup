"""선택 기능: 습득물 사진이 사용자 설명과 맞는지 멀티모달 모델로 판정한다."""

from __future__ import annotations

import base64
import urllib.error
import urllib.parse
import urllib.request

from langchain_core.exceptions import ModelInvalidRequestError

from .domain.models import LostItemQuery, SearchRecord, VisionAssessment


ALLOWED_IMAGE_HOSTS = {"minwon24.police.go.kr"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


class ImageDownloadError(RuntimeError):
    """비전 판정에 사용할 이미지를 안전하게 준비하지 못한 경우."""


def _download_image_as_data_url(
    image_url: str,
    *,
    timeout_seconds: float = 10,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> str:
    """허용된 경찰민원24 이미지를 내려받아 모델용 data URL로 변환한다."""
    parsed = urllib.parse.urlsplit(image_url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_IMAGE_HOSTS:
        raise ImageDownloadError("허용되지 않은 이미지 주소입니다.")

    request = urllib.request.Request(
        image_url,
        headers={
            "User-Agent": "jupjup/0.1",
            "Accept": "image/jpeg,image/png,image/webp",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get_content_type().lower()
            if content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise ImageDownloadError(f"지원하지 않는 이미지 형식입니다: {content_type}")

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ImageDownloadError("이미지 크기가 제한을 초과했습니다.")

            image_bytes = response.read(max_bytes + 1)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise ImageDownloadError("이미지를 내려받지 못했습니다.") from exc

    if not image_bytes or len(image_bytes) > max_bytes:
        raise ImageDownloadError("이미지가 비어 있거나 크기 제한을 초과했습니다.")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


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

        try:
            image_data_url = _download_image_as_data_url(record.image_url)
        except ImageDownloadError:
            # 사진 한 장의 장애가 전체 습득물 검색을 중단시키지 않게 한다.
            return None

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
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )
        try:
            result = self._model.invoke([message])
        except ModelInvalidRequestError:
            return None
        if isinstance(result, VisionAssessment):
            return result
        return VisionAssessment.model_validate(result)
