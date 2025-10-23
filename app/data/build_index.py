from app.embeddings_index import EmbeddingIndex
from app.config import EMBEDDING_MODEL

def main():
    idx = EmbeddingIndex(EMBEDDING_MODEL)
    idx.build_from_jsonl("app/data/examples.jsonl", save_index_path="app/data/faiss.index")
    print("✅ FAISS index built successfully!")

if __name__ == "__main__":
    main()
