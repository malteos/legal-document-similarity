import logging
import os
from collections import defaultdict
from typing import List

import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from docsim.methods import RecSys

stop_words = set(stopwords.words('english'))


logger = logging.getLogger(__name__)


class KeyedVectorRecSys(RecSys):
    # document ids should be str not int
    model = None  # type: KeyedVectors
    vectors_cls = KeyedVectors
    vector_size = 200

    def retrieve_recommendations(self, seed_doc_id: str):
        try:
            return [rec_doc_id for rec_doc_id, _ in self.model.most_similar(positive=seed_doc_id, topn=self.top_k)]
        except KeyError:
            logger.warning(f'Cannot retrieve recommendations, seed ID does not exist:{seed_doc_id}')
            return []

    def save_word2vec_format(self, file_path, override=False, **kwargs):
        if override and os.path.exists(file_path):
            logger.debug(f'Override {file_path}')
            os.remove(file_path)
            
        self.model.save_word2vec_format(file_path, **kwargs)

    def load_word2vec_format(self, file_path, **kwargs):
        self.model = KeyedVectors.load_word2vec_format(file_path, **kwargs)
        self.vector_size = self.model.vector_size

    def train(self, texts: List):
        raise NotImplementedError('KeyedVector requires pre-computed document vectors')


class MultiKeyedVectorRecSys(KeyedVectorRecSys):
    def train(self, items, **kwargs):
        # Load sub-models
        models = [KeyedVectors.load_word2vec_format(fp, **kwargs) for fp in items]

        self.vector_size = np.sum([m.vector_size for m in models])

        # Build new keyed vector model
        self.model = KeyedVectors(vector_size=self.vector_size)

        missing_docs = 0

        # Iterate over all words (in first model)
        for doc_id in models[0].index2word:
            # Stack vectors from all models
            models_vec = []
            for m in models:
                if doc_id in m.index2word:
                    models_vec.append(m.get_vector(doc_id))
                else:
                    # Use zero-vector if doc id does not exist
                    # print(f'WARNING: {doc_id} does not exist in {m}')
                    models_vec.append(np.zeros((m.vector_size)))
                    missing_docs += 1

            vec = np.hstack(models_vec)

            self.model.add(doc_id, vec)

        if missing_docs > 0:
            logger.warning(f'Missing documents: {missing_docs}')

        return self.model

    
class EnsembleKeyedVectorRecSys(KeyedVectorRecSys):
    sub_models = []
    strategy = 'max'
    
    def train(self, items, **kwargs):
        # Load sub-models
        self.sub_models = [KeyedVectors.load_word2vec_format(fp, **kwargs) for fp in items]

        self.vector_size = 0
        for m in self.sub_models:
             self.vector_size += m.vector_size
    
    def retrieve_recommendations(self, seed_doc_id: str):
        try:
            recs = defaultdict(float)
            
            # Retrieve recs for each submodel
            for model in self.sub_models:
                for rec_doc_id, score in model.most_similar(positive=seed_doc_id, topn=self.top_k):
                    recs[rec_doc_id] += score  # if multiple models return same rec, just add scores
            
            # Select recs with max score
            sorted_recs = [k for k, v in sorted(recs.items(), key=lambda item: item[1], reverse=True)]

            return sorted_recs[:self.top_k]

        except KeyError:
            logger.warning(f'Cannot retrieve recommendations, seed ID does not exist:{seed_doc_id}')
            return []