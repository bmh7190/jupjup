"""대화 이력에서 검색·신고서 의도와 보존할 정보를 판별한다."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .models import AgentResult, LostItemQuery, LostReportDraft


_REPORT_REQUEST_PATTERNS = (
    re.compile(
        r"(?:분실\s*)?신고(?:서|내용|문)?(?:를|을|도)?\s*"
        r".{0,12}(?:써|작성|정리|검토|점검|준비|만들|도와)"
    ),
    re.compile(
        r"(?:써|작성|정리|검토|점검|준비|만들|도와).{0,12}"
        r"(?:분실\s*)?신고(?:서|내용|문)?"
    ),
    re.compile(
        r"(?:민원\s*)?접수.{0,12}(?:도와|준비|작성|검토|점검|빠진|누락)"
    ),
    re.compile(r"(?:빠진|누락).{0,12}(?:내용|항목).{0,12}(?:봐|확인|검토)"),
)
_SEARCH_REQUEST_PATTERNS = (
    re.compile(r"(?:을|를)\s*(?:잃어버|분실(?:했|한|함)|두고|놓고)"),
    re.compile(r"(?:찾아|조회|검색)\s*(?:줘|해\s*줘|부탁)"),
    re.compile(r"(?:이야|예요|입니다).{0,40}(?:잃어버|분실(?:했|한|함))"),
)
_SEARCH_COMMAND_PATTERN = re.compile(
    r"(?:찾아|조회|검색)\s*(?:줘|해\s*줘|부탁)"
)
_SEARCH_CANCEL_PATTERN = re.compile(
    r"(?:찾아|조회|검색).{0,10}(?:하지\s*마|말아\s*줘|취소|그만)|"
    r"(?:하지\s*마|말아\s*줘|취소|그만).{0,10}(?:찾아|조회|검색)"
)
_REPORT_CANCEL_PATTERN = re.compile(
    r"(?:신고(?:서|내용|문)?\s*)?(?:작성\s*)?"
    r"(?:하지\s*마|말고|"
    r"그만(?:해|할|하고|둘|둘게|이야|할래|(?=[.!~\s]|$))|"
    r"취소(?:해|할|하고|할게|됐|(?=[.!~\s]|$))|"
    r"중단(?:해|할|하고|할게|(?=[.!~\s]|$)))"
)
_CONVERSATION_CLOSE_PATTERN = re.compile(
    r"^(?:고마워(?:요)?|감사(?:해요|합니다)?|됐어(?:요)?|괜찮아(?:요)?|"
    r"끝|종료|마무리(?:해\s*줘)?)(?:[.!~\s]*)$"
)
_REPORT_EDIT_PATTERN = re.compile(
    r"(?:수정|고쳐|바꿔|변경|정정|반영|추가|보완|삭제|빼\s*줘|"
    r"아니라|맞아|틀렸)"
)
_REPORT_DETAIL_PATTERN = re.compile(
    r"(?:\d{4}\s*[-년./]\s*\d{1,2}|\d{1,2}\s*월\s*\d{1,2}\s*일|"
    r"오늘|어제|그제|오전|오후|\d{1,2}\s*시|"
    r"\d+(?:\.\d+)?\s*(?:cm|mm|m|센티|개)|"
    r"(?:빨간|빨강|파란|파랑|검정|검은|하얀|흰색|노란|노랑|초록|"
    r"갈색|회색|보라|분홍|주황)|"
    r"(?:역|출구|정류장|공항|터미널|매장|카페|식당|학교|공원|도로|"
    r"건물|엘리베이터|화장실|주차장)(?:에서|근처|앞|안|이고|이야|예요)?)"
)
_QUESTION_OR_UNRELATED_REQUEST_PATTERN = re.compile(
    r"(?:\?|어때(?:요)?|뭐(?:야|지|가)|왜|어떻게|알려\s*줘|설명\s*해|추천\s*해)"
)
_REPORT_CONTEXT_FIELDS = (
    "item_name",
    "category",
    "lost_date",
    "lost_time",
    "lost_place",
    "region",
    "color",
    "size",
    "brand",
    "quantity",
    "features",
    "circumstances",
)
_GENERIC_ITEM_NAMES = {
    "",
    "물건",
    "분실물",
    "습득물",
    "그거",
    "그것",
    "뭐",
    "무언가",
}


def message_content_text(content: Any) -> str:
    """문자열 또는 멀티모달 메시지에서 텍스트만 꺼낸다."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for block in content:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text", "")))
    return "\n".join(chunk for chunk in chunks if chunk)


def latest_user_text(messages: list[Any]) -> str:
    """Memory 전체에서 현재 턴의 마지막 사용자 메시지를 찾는다."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message_content_text(message.content)
        if isinstance(message, dict) and message.get("role") == "user":
            return message_content_text(message.get("content"))
    return ""


def is_search_cancel_request(text: str) -> bool:
    return bool(_SEARCH_CANCEL_PATTERN.search(text))


def is_explicit_report_request(text: str) -> bool:
    """사용자가 현재 메시지에서 신고서 도움을 분명히 요청했는지 판별한다."""
    normalized = re.sub(r"\s+", " ", text.strip())
    if _REPORT_CANCEL_PATTERN.search(normalized):
        return False
    return any(pattern.search(normalized) for pattern in _REPORT_REQUEST_PATTERNS)


def is_explicit_search_request(text: str) -> bool:
    """물품을 명시해 바로 조회해 달라는 현재 턴의 표현을 판별한다."""
    if is_explicit_report_request(text):
        return False
    normalized = re.sub(r"\s+", " ", text.strip())
    return any(pattern.search(normalized) for pattern in _SEARCH_REQUEST_PATTERNS)


def _human_texts(messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            texts.append(message_content_text(message.content))
        elif isinstance(message, dict) and message.get("role") == "user":
            texts.append(message_content_text(message.get("content")))
    return texts


def _is_report_workflow_stop(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip())
    return bool(
        _REPORT_CANCEL_PATTERN.search(normalized)
        or _CONVERSATION_CLOSE_PATTERN.fullmatch(normalized)
        or (
            _SEARCH_COMMAND_PATTERN.search(normalized)
            and not is_explicit_report_request(normalized)
        )
    )


def _assistant_requested_report_details(messages: list[Any]) -> bool:
    """직전 답변이 신고서 보완 정보를 물었는지 확인한다."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            continue
        if isinstance(message, AIMessage):
            text = message_content_text(message.content)
            return bool(
                re.search(
                    r"(?:신고서|초안|분실).{0,80}(?:알려|확인|필요|무엇|언제|어디|어떤)|"
                    r"(?:알려|확인|필요|무엇|언제|어디|어떤).{0,80}(?:신고서|초안|분실)",
                    text,
                )
            )
    return False


def _is_report_follow_up(text: str, messages: list[Any]) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized or _is_report_workflow_stop(normalized):
        return False
    if _REPORT_EDIT_PATTERN.search(normalized):
        return True
    if _QUESTION_OR_UNRELATED_REQUEST_PATTERN.search(normalized):
        return False
    if _REPORT_DETAIL_PATTERN.search(normalized):
        return True
    return bool(_assistant_requested_report_details(messages) and len(normalized) <= 80)


def is_report_workflow_turn(messages: list[Any]) -> bool:
    """현재 턴이 사용자가 시작한 신고서 작성 흐름에 속하는지 판별한다."""
    human_texts = _human_texts(messages)
    if not human_texts:
        return False

    active = False
    for previous_text in human_texts[:-1]:
        if is_explicit_report_request(previous_text):
            active = True
        elif active and _is_report_workflow_stop(previous_text):
            active = False

    current_text = human_texts[-1]
    if is_explicit_report_request(current_text):
        return True
    if not active:
        return False
    return _is_report_follow_up(current_text, messages[:-1])


def _item_name_from_text(text: str) -> str | None:
    """검색 강제 여부에만 쓰는 보수적인 물품명 단서를 찾는다."""
    normalized = re.sub(r"\s+", " ", text.strip())
    patterns = (
        re.compile(
            r"([가-힣A-Za-z0-9][가-힣A-Za-z0-9 ]{0,30}?)(?:을|를)\s*"
            r"(?:잃어버|분실(?:했|한|함)|두고|놓고|찾아|조회|검색)"
        ),
        re.compile(
            r"([가-힣A-Za-z0-9][가-힣A-Za-z0-9 ]{0,20}?)\s*"
            r"(?:이야|야|예요|입니다)(?:[.!\s]|$)"
        ),
        re.compile(
            r"([가-힣A-Za-z0-9][가-힣A-Za-z0-9 ]{0,20}?)\s+"
            r"(?:찾아|조회|검색)\s*(?:줘|해\s*줘|부탁)"
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            candidate = re.split(r"(?:에서|에게|에는|은|는)\s+", match.group(1))[-1]
            if re.search(r"(?:가상\s*사례|기능\s*검증용?\s*대화)", candidate):
                continue
            candidate = re.sub(r"^(?:같은 조건|방금 말한)\s*", "", candidate).strip()
            words = [re.sub(r"(?:만|도)$", "", word) for word in candidate.split()]
            while words and words[-1] in _GENERIC_ITEM_NAMES:
                words.pop()
            if words:
                return words[-1]
    return None


def _tool_artifact_source(
    message: Any,
) -> LostItemQuery | LostReportDraft | None:
    if not isinstance(message, ToolMessage):
        return None
    artifact = message.artifact
    try:
        if message.name == "search_lost112_candidates":
            result = (
                artifact
                if isinstance(artifact, AgentResult)
                else AgentResult.model_validate(artifact)
            )
            return result.query
        if message.name == "prepare_lost_report_draft":
            return (
                artifact
                if isinstance(artifact, LostReportDraft)
                else LostReportDraft.model_validate(artifact)
            )
    except (TypeError, ValueError):
        return None
    return None


def _previous_item_name(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        source = _tool_artifact_source(message)
        if source is not None and source.item_name:
            return source.item_name
    return None


def has_search_item_context(messages: list[Any]) -> bool:
    """현재 문장 또는 같은 thread의 이전 Tool에 물품명이 있는지 확인한다."""
    return bool(_item_name_from_text(latest_user_text(messages)) or _previous_item_name(messages))


def report_context_from_messages(messages: list[Any]) -> dict[str, Any]:
    """이전 검색·신고 Tool artifact에서 사용자가 확인한 신고 정보를 모은다."""
    context: dict[str, Any] = {}
    successful_calls = {
        (message.tool_call_id, message.name)
        for message in messages
        if isinstance(message, ToolMessage)
        and _tool_artifact_source(message) is not None
    }

    def apply_values(values: dict[str, Any], *, explicit: bool) -> None:
        item_name = values.get("item_name")
        previous_item_name = context.get("item_name")
        if (
            isinstance(item_name, str)
            and item_name.strip()
            and isinstance(previous_item_name, str)
            and previous_item_name.strip()
            and item_name.strip().casefold() != previous_item_name.strip().casefold()
        ):
            context.clear()
        for field in _REPORT_CONTEXT_FIELDS:
            if field not in values:
                continue
            value = values[field]
            if explicit or value not in (None, "", []):
                context[field] = value

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call.get("name") not in {
                    "search_lost112_candidates",
                    "prepare_lost_report_draft",
                }:
                    continue
                if (call.get("id"), call.get("name")) not in successful_calls:
                    continue
                args = call.get("args", {})
                if isinstance(args, dict):
                    apply_values(args, explicit=True)
        source = _tool_artifact_source(message)
        if source is not None:
            apply_values(
                {field: getattr(source, field, None) for field in _REPORT_CONTEXT_FIELDS},
                explicit=False,
            )
    return context


def search_tool_called_since_latest_user(messages: list[Any]) -> bool:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return False
        if (
            isinstance(message, ToolMessage)
            and message.name == "search_lost112_candidates"
        ):
            return True
    return False
