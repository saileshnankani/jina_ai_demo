# Jina Faiss Retrieval

Modified example from https://gist.github.com/tadejsv/32091353449c301c6506f42e70410809 to use Jina 2.0 with Faiss on sample [MS Marco data](https://microsoft.github.io/msmarco/)

## Installation

```
conda create -n jina-2.0 -c conda-forge -c huggingface faiss-cpu datasets
conda activate jina-2.0
pip install jina sentence-transformers
```