# AI 기술 면접 RAG 시스템

[boost-devs/ai-tech-interview](https://github.com/boost-devs/ai-tech-interview) 레포를 데이터 소스로 활용한 RAG(Retrieval-Augmented Generation) 시스템입니다.  
사용자의 질문을 분석해 카테고리를 분류하고, 관련 면접 Q&A를 검색·재순위화한 뒤 GPT-4o-mini로 답변을 생성합니다.

---

## 파이프라인 개요

```
[사용자 질문]
     ↓
[쿼리 분석 (GPT-4o-mini)]  ← 카테고리 분류 + 면접 스타일 재작성
     ↓
[벡터 검색 (ChromaDB + BGE-M3)]  ← 카테고리 필터 적용
     ↓
[재순위 (BGE-Reranker-Base)]
     ↓
[답변 생성 (GPT-4o-mini)]
```

---

## 프로젝트 구조

```
interview_data/
├── main.py                  # CLI 진입점
├── requirements.txt
├── src/
│   ├── ingest.py            # 마크다운 파싱 → JSON 변환
│   ├── embed.py             # 임베딩 생성 → ChromaDB 저장
│   ├── retrieve.py          # 벡터 검색
│   ├── rerank.py            # BGE 재순위
│   ├── generate.py          # 전체 RAG 파이프라인
│   ├── evaluate.py          # 검색 품질 평가
│   └── test_rewrite.py      # 쿼리 재작성 테스트
├── data/
│   ├── raw/                 # ai-tech-interview 원본 마크다운
│   └── processed/
│       └── qa.json          # 파싱된 Q&A 데이터
└── chroma_db/               # 벡터 DB 저장 디렉토리
```

---

## 카테고리

| 카테고리 | 주요 주제 |
|---|---|
| `statistics-math` | 통계, 확률, 수학, 분포, 가설검정 |
| `machine-learning` | ML 알고리즘, overfitting, 정규화, 앙상블 |
| `deep-learning` | 신경망, gradient descent, dropout, activation function |
| `python` | GIL, 자료형, 라이브러리, 문법 |
| `network` | TCP/UDP, HTTP/HTTPS, 네트워크 프로토콜 |
| `operating-system` | 프로세스, 스레드, 메모리, 데드락 |
| `data-structure` | 스택, 큐, 트리, 해시 |
| `algorithm` | 정렬, 탐색, DP, 그래프 |

---

## 환경 세팅

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY 입력
```

---

## 데이터 준비 및 인덱싱

```bash
# 1. 마크다운 파싱 → data/processed/qa.json 생성
python -m src.ingest

# 2. 임베딩 생성 → chroma_db/ 저장
python -m src.embed
```

---

## 실행

`main.py`는 `--query` 인자 하나만 받는 CLI 스크립트입니다.  
내부적으로 `src/generate.py`의 `generate_answer()`를 호출해 전체 RAG 파이프라인을 실행합니다.

```bash
python main.py --query "<질문>"
```

**예시**

```bash
python main.py --query "TCP와 UDP의 차이는?"
python main.py --query "overfitting 방지 방법"
python main.py --query "GIL이란 무엇인가요?"
python main.py --query "gradient descent란?"
```

**출력 예시**

```
질문: overfitting 방지 방법
==================================================
쿼리 분석 중...
카테고리: machine-learning
재작성된 쿼리: 오버피팅을 방지하기 위한 방법에는 어떤 것들이 있나요?
검색 결과: 5개
재순위 후: 3개
  [1] 오버피팅이란 무엇이고, 어떻게 방지할 수 있나요? | rerank_score: 4.1234
  ...
(GPT-4o-mini 생성 답변)
==================================================
```

> `--query` 는 필수 인자입니다. 생략하면 오류가 발생합니다.

---

## 모듈 설명

### `src/ingest.py`
`data/raw/ai-tech-interview/answers/` 내 마크다운 파일을 파싱하여 `{category, question, answer, source_file}` 형태의 JSON으로 저장합니다. `###` / `####` 헤더를 질문으로, 그 아래 본문을 답변으로 추출합니다.

### `src/embed.py`
`BAAI/bge-m3` 모델로 질문 텍스트를 임베딩하고 ChromaDB(코사인 유사도)에 저장합니다.

### `src/retrieve.py`
쿼리를 임베딩하여 ChromaDB에서 유사 Q&A를 검색합니다. 카테고리 필터와 유사도 임계값(threshold)을 지원합니다.

### `src/rerank.py`
`BAAI/bge-reranker-base`로 검색 결과를 재순위화합니다. 모델은 싱글톤으로 관리되어 중복 로딩을 방지합니다.

### `src/generate.py`
전체 RAG 파이프라인의 진입점입니다. 쿼리 분석 → 검색 → 재순위 → GPT-4o-mini 답변 생성을 순서대로 실행합니다.

### `src/evaluate.py`
12개 테스트 쿼리에 대해 Top-1 검색 결과의 키워드 히트 여부와 유사도 점수를 출력합니다.

### `src/test_rewrite.py`
GPT-4o-mini 기반 쿼리 재작성 기능을 독립적으로 테스트합니다.

---

## 기술 스택

- **임베딩**: `sentence-transformers` (BAAI/bge-m3)
- **벡터 DB**: `chromadb`
- **재순위**: `FlagEmbedding` (BAAI/bge-reranker-base)
- **LLM**: `openai` (gpt-4o-mini)
- **환경 변수**: `python-dotenv`
