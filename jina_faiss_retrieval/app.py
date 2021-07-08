import os
import sys
import argparse
from jina import Flow, Document
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


def document_generator(file_name):
    """
    Define data as Document to be indexed.
    """
    import csv
    notebook_path = os.path.abspath("trial.ipynb")
    data_path = os.path.join(os.path.dirname(notebook_path), file_name)

    # Get Document and ID
    with open(data_path, newline="") as f:
        reader = csv.reader(f, delimiter='\t')
        for data in reader:
            d = Document()
            # docid
            d.tags['id'] = int(data[0])
            # doc
            d.text = data[1]
            yield d


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
            f.index(document_generator('collection.short.tsv'), batch_size=30, show_progress=True)
        if args.search:
            f.search(document_generator('queries.short.tsv'), batch_size=30, show_progress=True)
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
