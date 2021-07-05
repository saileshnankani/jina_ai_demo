import os
import urllib.request
import webbrowser
from pathlib import Path

from jina import Flow, Document
from jina.importer import ImportExtensions
from jina.logging.predefined import default_logger
from jina.logging.profile import ProgressBar
from jina.parsers.helloworld import set_hw_chatbot_parser
from jina.types.document.generators import from_csv

if __name__ == '__main__':
    from my_executors import TransformerEmbed, FaissIndexer
else:
    from .my_executors import TransformerEmbed, FaissIndexer


def _get_flow(args):
    """Ensure the same flow is used in hello world example and system test."""
    return (
        Flow(cors=True)
        .add(name="encoder", uses=TransformerEmbed, parallel=2)
        .add(name="indexer", uses=FaissIndexer, parallel=6, workspace=args.workdir)
    )


def index_generator():
    """
    Define data as Document to be indexed.
    """
    import csv
    notebook_path = os.path.abspath("trial.ipynb")
    data_path = os.path.join(os.path.dirname(notebook_path), 'collection.short.tsv')

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


def query_generator():
    """
    Define data as Document to be indexed.
    """
    import csv
    notebook_path = os.path.abspath("trial.ipynb")
    data_path = os.path.join(os.path.dirname(notebook_path), 'queries.short.tsv')

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


def hello_world(args):
    """
    Execute the chatbot example.

    :param args: arguments passed from CLI
    """
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    with ImportExtensions(
        required=True,
        help_text='this demo requires Pytorch and Transformers to be installed, '
        'if you haven\'t, please do `pip install jina[torch,transformers]`',
    ):
        import transformers
        import torch

        assert [torch, transformers]  #: prevent pycharm auto remove the above line

    f = _get_flow(args)
    f.plot()

    # index it!
    with f:
        f.index(index_generator, batch_size=8, show_progress=True)
        f.search(query_generator, batch_size=8, show_progress=True)
        return


if __name__ == '__main__':
    args = set_hw_chatbot_parser().parse_args()
    hello_world(args)
