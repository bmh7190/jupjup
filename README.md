# JupJup (줍줍)

사용자의 자연어에서 분실물 정보를 추출하고 경찰청 Open API 3종을 조회해 가능성이 높은 습득물 후보를 추천하는 1차 구현입니다.

## 이번 구현 범위

1. 자연어에서 물품명, 날짜, 장소, 색상, 특징 추출
2. 빠진 정보에 대한 추가 질문 생성
3. 경찰청 분실물정보 조회
4. 경찰청 습득물정보와 포털기관 습득물정보 조회
5. 물품명, 분류, 색상, 장소, 날짜, 특징을 이용한 설명 가능한 유사도 계산

경찰민원24 신고서 작성과 접수 연결은 후속 기획 범위입니다.

## 구조

```text
src/jupjup/
├── config.py       # .env와 실행 설정
├── models.py       # Pydantic 데이터 모델
├── privacy.py      # LLM 전달 전 개인정보 마스킹
├── extractor.py    # LangChain 구조화 정보 추출과 누락 질문
├── api_client.py   # 경찰청 API 3종 조회와 XML 표준화
├── matcher.py      # 유사도 계산과 추천 근거
├── vision.py       # 선택적인 습득물 사진 판정
├── service.py      # 전체 실행 순서
├── cli.py          # 터미널 실행 화면
└── demo.py         # API 키 없는 로컬 데모
```

`경찰청 분실물정보`는 다른 사용자의 분실 신고이므로 습득물 후보와 섞지 않고 `유사한 기존 분실 신고`로 표시합니다. 실제 후보 추천에는 `경찰청 습득물정보`와 `포털기관 습득물정보`를 사용합니다.

## 설치

```bash
cd /Users/bmh7190/skala/skala-langchain/jupjup
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

`.env`에 키를 입력합니다.

```dotenv
DATA_GO_KR_SERVICE_KEY=공공데이터포털_인증키
OPENAI_API_KEY=OpenAI_API_키
OPENAI_MODEL=gpt-4.1-mini
```

기존 `.env`가 있다면 프로젝트 루트로 옮기거나 `--env` 뒤에 해당 파일의 절대경로를 지정할 수 있습니다. 자연어 추출에는 `OPENAI_API_KEY`가 필요합니다.

## 실행

외부 API 없이 구조와 점수를 먼저 확인합니다.

```bash
PYTHONPATH=src python -m jupjup.cli --demo
```

실제 대화형 실행:

```bash
jupjup
```

한 문장으로 실행:

```bash
jupjup --text "어제 저녁 강남역에서 검은색 카드지갑을 잃어버렸어"
```

등록된 습득물 사진을 멀티모달 모델로 추가 평가하려면 `--vision`을 붙입니다. 사진이 없는 후보에는 적용되지 않으며 OpenAI API 비용이 발생합니다.

```bash
jupjup --vision
```

## 점수 기준

기본 가중치는 물품명 30%, 장소 20%, 색상 15%, 날짜 15%, 분류 10%, 특징 10%입니다. 사용자에게서 얻지 못한 항목은 점수에서 제외하고 남은 항목의 가중치를 다시 합산합니다. 따라서 빈 값을 불일치로 잘못 계산하지 않습니다.

사진 판정을 켜면 상위 3개 후보 중 실제 사진이 있는 항목에만 15% 가중치를 추가합니다. 최종 화면에는 총점과 함께 `날짜 유사도가 높음`, `확인 가능한 습득물 사진이 있음` 같은 근거가 표시됩니다.

## 테스트

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
