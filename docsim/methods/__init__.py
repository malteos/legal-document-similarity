import logging
import os
import pickle
from typing import List

from nltk.corpus import stopwords
from tqdm import tqdm, tqdm_notebook

stop_words = set(stopwords.words('english'))


logger = logging.getLogger(__name__)


class RecSys(object):
    name = 'unnamed_recsys'
    model = None
    print_progress = False
    tqdm_notebook = False

    def __init__(self, doc_id2idx, idx2doc_id, top_k=10, **kwargs):
        self.doc_id2idx = doc_id2idx
        self.idx2doc_id = idx2doc_id
        self.top_k = top_k

        # Set all other kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def get_progress_iterator(self, iterator, **kwargs):
        if self.print_progress:
            if self.tqdm_notebook:
                iterator = tqdm_notebook(iterator, **kwargs)
            else:
                iterator = tqdm(iterator, **kwargs)

        return iterator

    def train(self, items: List):
        """

        :param items: List of items (document texts, citations, ...) ordered by indexes
        :return:
        """
        raise NotImplementedError()

    def retrieve_recommendations(self, seed_doc_id: int) -> List[int]:
        """

        :param seed_doc_id: CL opinion ID of the seed document
        :return: List of recommended CL opinion IDs
        """
        raise NotImplementedError()

    def before_save_to_disk(self):
        pass

    def after_save_to_disk(self):
        pass

    def save_to_disk(self, file_path, override=False):
        if not override and os.path.exists(file_path):
            # raise FileExistsError(f'File exists already: {file_path}')
            logger.warning(f'File exists already: {file_path}')
            return

        self.before_save_to_disk()

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

        self.after_save_to_disk()

    @classmethod
    def load_from_disk(cls, file_path, **kwargs):
        with open(file_path, 'rb') as f:
            rs = pickle.load(f)

            for k, v in kwargs.items():
                if hasattr(rs, k):
                    setattr(rs, k, v)

            return rs
