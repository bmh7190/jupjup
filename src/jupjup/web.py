"""터미널 전용이던 채팅을 브라우저에서 확인할 수 있게 하는 최소 가시화 서버.

기존 Agent 로직(agent.py, service.py 등)은 전혀 바꾸지 않고,
`JupJupChatAgent.chat()` 호출 결과를 그대로 JSON으로 감싸서 보여주기만 한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import JupJupChatAgent, JupJupChatResponse
from .api_client import Lost112ApiClient
from .config import Settings
from .demo import run_demo
from .service import JupJupAgentService

logger = logging.getLogger(__name__)

app = FastAPI(title="JupJup 채팅 가시화")

_agent: JupJupChatAgent | None = None
_init_error: str | None = None
_init_attempted = False


def _build_agent() -> JupJupChatAgent:
    settings = Settings.from_env()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")

    api_client = Lost112ApiClient(
        settings.data_service_key,
        **settings.api_client_options,
    )
    # cli.py의 기본 동작과 동일하게 vision은 기본적으로 끄고, `--vision`에 해당하는
    # 명시적 옵션 없이는 켜지 않는다. (사진 URL을 OpenAI 서버가 못 받아오면
    # 검색 자체가 400 오류로 실패하기 때문에 기본값으로 켜두면 위험하다.)
    service = JupJupAgentService(api_client)
    return JupJupChatAgent(service, model_name=settings.openai_model, api_key=settings.openai_api_key)


def _get_agent() -> JupJupChatAgent | None:
    """최초 요청 시 한 번만 Agent를 만들고, 실패 원인은 화면에 그대로 보여준다."""
    global _agent, _init_error, _init_attempted
    if not _init_attempted:
        _init_attempted = True
        try:
            _agent = _build_agent()
        except Exception as exc:  # noqa: BLE001 - 원인을 데모 화면에 그대로 노출
            _init_error = str(exc)
            logger.warning("Agent 초기화 실패: %s", exc)
    return _agent


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatReply(BaseModel):
    thread_id: str
    response: JupJupChatResponse


@app.get("/", response_class=HTMLResponse)
def index() -> Response:
    html_path = Path(__file__).parent / "web_assets" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    # 개발 중 화면이 자주 바뀌므로 브라우저가 이전 버전을 캐시해 보여주지 않게 한다.
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon() -> Response:
    # 브라우저가 탭 아이콘을 자동으로 요청하는데, 우리가 아이콘을 안 두면
    # 매번 404 로그만 남는다. 별도 아이콘 없이 조용히 204로 응답한다.
    return Response(status_code=204)


@app.get("/api/status")
def status() -> dict:
    agent = _get_agent()
    return {"ready": agent is not None, "error": _init_error}


@app.post("/api/chat", response_model=ChatReply)
def chat(payload: ChatRequest) -> ChatReply:
    thread_id = payload.thread_id or str(uuid4())
    text = payload.message.strip()
    if not text:
        return ChatReply(
            thread_id=thread_id,
            response=JupJupChatResponse(message="메시지를 입력해주세요."),
        )

    agent = _get_agent()
    if agent is None:
        message = (
            f"실행 오류: {_init_error}\n"
            "실제 API 키 없이 화면만 확인하려면 '데모 모드'를 사용해보세요."
        )
        return ChatReply(thread_id=thread_id, response=JupJupChatResponse(message=message))

    try:
        response = agent.chat(text, thread_id=thread_id)
    except Exception as exc:  # noqa: BLE001 - 서버가 죽지 않고 오류를 화면에 보여준다.
        # 외부 SDK 예외에는 요청 헤더가 포함될 수 있으므로 스택과 원문을
        # 배포 로그나 사용자 응답에 남기지 않는다.
        logger.error("Agent 실행 중 오류 (%s)", type(exc).__name__)
        response = JupJupChatResponse(
            message="실행 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )

    return ChatReply(thread_id=thread_id, response=response)


@app.post("/api/demo", response_model=ChatReply)
def demo(payload: ChatRequest) -> ChatReply:
    """API 키 없이도 후보 카드 화면을 바로 확인할 수 있는 데모 모드."""
    thread_id = payload.thread_id or str(uuid4())
    result = run_demo()
    response = JupJupChatResponse(
        message="(데모 모드) 예시 데이터로 조회한 결과입니다. 실제 API 호출은 하지 않았습니다.",
        search_result=result,
    )
    return ChatReply(thread_id=thread_id, response=response)


def run() -> None:
    """`jupjup-web` 콘솔 스크립트 진입점."""
    import uvicorn

    uvicorn.run("jupjup.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
