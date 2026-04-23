from FlagEmbedding import FlagReranker

RERANKER_MODEL = "BAAI/bge-reranker-base"

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"리랭커 모델 로딩 중: {RERANKER_MODEL}")
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
    return _reranker


def rerank(query: str, results: list[dict], top_k: int = 3) -> list[dict]:
    if not results:
        return results

    reranker = get_reranker()

    # 쿼리-답변 쌍 생성
    pairs = [[query, r["question"]] for r in results]

    # 점수 계산
    scores = reranker.compute_score(pairs)

    # 점수 붙이기
    for i, result in enumerate(results):
        result["rerank_score"] = round(float(scores[i]), 4)

    # 재정렬
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]