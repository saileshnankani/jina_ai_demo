import faiss
import os
from datasets import load_dataset
from jina import Document, DocumentArray, Executor, Flow, requests
from sentence_transformers import SentenceTransformer
from pathlib import Path


class TransformerEmbed(Executor):  # Embedd text using transformers
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

    @requests
    def embedd(self, docs: DocumentArray, **kwargs):
        for d in docs:
            d.embedding = self.model.encode([d.text])  # list as faiss needs 2d arrays


class FaissIndexer(Executor):  # Simple exact FAISS indexer
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._docs = DocumentArray()
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(384)
        self._index = faiss.index_cpu_to_all_gpus(index_flat)
        Path(self.workspace).mkdir(parents=True, exist_ok=True)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._docs.extend(docs)
        _ = [self._index.add(d.embedding) for d in docs]
        index_file = os.path.join(self.workspace, "index_file")
        faiss.write_index(self._index, index_file)
        self._docs.save('data.bin', file_format='binary')

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        index_file = os.path.join(self.workspace, "index_file")
        print("index_file: ", index_file)
        index = faiss.read_index(index_file)
        self._docs = DocumentArray.load('data.bin', file_format='binary')
        with open("myfile.txt", "w") as f:
            for doc in docs:
                dists, matches = index.search(doc.embedding, 10)  # top 10 matches
                rank = 1
                for d, i in zip(dists[0], matches[0]):
                    doc_copy = Document(self._docs[int(i)], copy=True)
                    doc_copy.score = d
                    doc.matches.append(doc_copy)
                    f.write(f"{int(doc.tags['id'])}\t{int(doc_copy.tags['id'])}\t{rank}\n")
                    rank += 1
