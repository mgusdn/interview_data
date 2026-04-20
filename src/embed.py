import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/processed/qa.json")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "interview_qa"
MODEL_NAME = "BAAI/bge-m3"

def main():
    # 1. 데이터 로드
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    print(f"총 {len(data)}개 Q&A 로드 완료")

    # 2. 임베딩 모델 로드
    print(f"모델 로딩 중: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Chroma DB 초기화
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # 기존 컬렉션 있으면 삭제 후 재생성
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("기존 컬렉션 삭제")
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
    )
    print("Chroma 컬렉션 생성 완료")

    # 4. 임베딩 생성 (질문 + 답변 전체를 임베딩)
    texts = [f"{item['question']} {item['answer']}" for item in data]
    print("임베딩 생성 중... (시간이 걸릴 수 있어요)")
    embeddings = model.encode(texts, show_progress_bar=True)

    # 5. Chroma에 저장
    collection.add(
        ids=[str(i) for i in range(len(data))],
        embeddings=embeddings.tolist(),
        documents=[item["answer"] for item in data],
        metadatas=[{
            "category": item["category"],
            "question": item["question"],
            "source_file": item["source_file"],
        } for item in data]
    )

    print(f"\n저장 완료! 총 {collection.count()}개 벡터 저장됨")


if __name__ == "__main__":
    main()