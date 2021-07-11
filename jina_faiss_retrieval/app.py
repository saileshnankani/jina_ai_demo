import os
import sys
import argparse
from jina import Flow, Document, DocumentArrayMemmap
from jina.importer import ImportExtensions
from pathlib import Path

if __name__ == '__main__':
    from my_executors import TransformerEmbed, FaissIndexer
else:
    from .my_executors import TransformerEmbed, FaissIndexer


def _get_flow(index_dir):
    """Ensure the same flow is used in hello world example and system test."""
    return (
        Flow(cors=True)
        .add(name="encoder", parallel=4, uses=TransformerEmbed)
        .add(name="indexer", parallel=4, uses=FaissIndexer,  workspace=index_dir)
    )


def run_retrieval(index_dir, args):
    """
    Runs the retrieval using Faiss
    """
    with ImportExtensions(
        required=True,
        help_text='this demo requires Pytorch and Transformers to be installed, '
        'if you haven\'t, please do `pip install jina[torch,transformers]`',
    ):
        import transformers
        import torch

        assert [torch, transformers]  #: prevent pycharm auto remove the above line

    f = _get_flow(index_dir)
    f.plot()

    # index it!
    with f:
        if args.index:
            corpus_documents = DocumentArrayMemmap('collection.short.tsv')
            f.index(corpus_documents, batch_size=30, show_progress=True)
        if args.search:
            query_documents = DocumentArrayMemmap('queries.short.tsv')
            f.search(query_documents, batch_size=30, show_progress=True)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', action='store_true', help='index files')
    parser.add_argument('--search', action='store_true', help='search corpus')
    args = parser.parse_args()
    cwd = os.getcwd()
    index_dir = os.path.join(cwd, 'index')
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    run_retrieval(index_dir, args)
