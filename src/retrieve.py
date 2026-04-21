import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "interview_qa"
MODEL_NAME = "BAAI/bge-m3"

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)

def retrieve(query: str, k: int = 5, category: str = None, threshold: float = 0.5) -> list[dict]:
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()

    query_embedding = model.encode([query])[0].tolist()

    where = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results["ids"][0])):
        score = round(1 - results["distances"][0][i], 4)
        
        # threshold 이하 제외
        if score < threshold:
            continue

        output.append({
            "rank": i + 1,
            "category": results["metadatas"][0][i]["category"],
            "question": results["metadatas"][0][i]["question"],
            "answer": results["documents"][0][i],
            "score": score,
        })

    return output


if __name__ == "__main__":
    print("=== 카테고리 필터 없이 검색 ===")
    results = retrieve("PCA란 무엇인가요?")
    for r in results:
        print(f"[{r['rank']}] ({r['category']}) {r['question']} | score: {r['score']}")
    print(f"→ {len(results)}개 반환\n")

    print("=== ML 카테고리만 검색 ===")
    results = retrieve("PCA 차원 축소", category="machine-learning")
    for r in results:
        print(f"[{r['rank']}] ({r['category']}) {r['question']} | score: {r['score']}")
    print(f"→ {len(results)}개 반환\n")

    print("=== Network 카테고리만 검색 ===")
    results = retrieve("TCP와 UDP의 차이", category="network")
    for r in results:
        print(f"[{r['rank']}] ({r['category']}) {r['question']} | score: {r['score']}")
    print(f"→ {len(results)}개 반환")