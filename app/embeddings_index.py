from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from pathlib import Path

class EmbeddingIndex:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.examples = []

    def build_from_jsonl(self, path_jsonl: str, save_index_path: str = None):
        # Open file and skip empty lines
        with open(path_jsonl, 'r', encoding='utf8') as f:
            lines = [json.loads(l) for l in f if l.strip()]

        texts = [l['text'] for l in lines]
        labels = [l['label'] for l in lines]

        # Create embeddings
        emb = np.array(self.model.encode(texts, show_progress_bar=True), dtype='float32')
        faiss.normalize_L2(emb)

        # Build FAISS index
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

        # Store examples
        self.examples = [{'text': t, 'label': lab} for t, lab in zip(texts, labels)]

        # Save index if path is provided
        if save_index_path:
            faiss.write_index(self.index, save_index_path)
            Path(save_index_path + ".meta").write_text(json.dumps(self.examples))
        
        print(f"✅ Built index with {len(self.examples)} entries")

    def load(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)
        self.examples = json.loads(Path(meta_path).read_text())

    def query(self, text: str, top_k: int = 3):
        emb = np.array(self.model.encode([text]), dtype='float32')
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            ex = self.examples[idx]
            results.append({'text': ex['text'], 'label': ex['label'], 'score': float(score)})
        return results
