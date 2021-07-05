import faiss
from datasets import load_dataset
from jina import Document, DocumentArray, Executor, Flow, requests
from sentence_transformers import SentenceTransformer


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
        self._index = faiss.IndexFlatL2(384)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._docs.extend(docs)
        _ = [self._index.add(d.embedding) for d in docs]

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        with open("myfile.txt", "w") as f:
            for doc in docs:
                dists, matches = self._index.search(doc.embedding, 10)  # top 10 matches
                rank = 1
                for d, i in zip(dists[0], matches[0]):
                    doc_copy = Document(self._docs[int(i)], copy=True)
                    doc_copy.score = d
                    doc.matches.append(doc_copy)
                    f.write(f"{int(doc.tags['id'])}\t{int(d.tags['id'])}\t{rank}\n")
                    rank += 1
