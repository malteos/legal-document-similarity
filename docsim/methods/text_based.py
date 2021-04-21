import logging
import os
import multiprocessing
from abc import ABC
from typing import List

import gensim
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from gensim.models.keyedvectors import Doc2VecKeyedVectors
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

from docsim.methods import RecSys
from docsim.methods.keyed_vector_based import KeyedVectorRecSys

stop_words = set(stopwords.words('english'))


logger = logging.getLogger(__name__)


class TextBasedRecSys(RecSys, ABC):
    def train(self, items: List):
        return self.train_from_texts(items)

    def train_from_texts(self, texts: List):
        raise NotImplementedError()


class TfIdfRecSys(TextBasedRecSys):
    stop_words = None
    vector_size = 10000

    def retrieve_recommendations(self, seed_doc_id: str) -> List[str]:
        if seed_doc_id not in self.doc_id2idx:
            raise ValueError(f'Seed ID not found in index: {seed_doc_id}')

        seed_idx = self.doc_id2idx[seed_doc_id]
        seed_vec = self.model[seed_idx]

        # TODO bool term filter first for speed up
        # TODO https://github.com/spotify/annoy (only for dense vectors)
        # use gensim? https://gist.github.com/clemsos/7692685
        cosine_similarities = linear_kernel(seed_vec,
                                            self.model).flatten()  # TODO compare against opinions published before seed

        related_idxs = cosine_similarities.argsort()[-self.top_k - 1:-1]
        # related_idxs = cosine_similarities.argsort()[:-self.top_k:-1]

        related_ids = [self.idx2doc_id[idx] for idx in related_idxs]

        return related_ids

    def train_from_texts(self, texts: List):
        self.model = TfidfVectorizer(stop_words=self.stop_words, max_features=self.vector_size).fit_transform(texts)

        return self.model


class Doc2VecRecSys(KeyedVectorRecSys):
    model = None  # type: Doc2VecKeyedVectors
    window = 5
    vectors_cls = Doc2VecKeyedVectors

    def train(self, texts: List):
        logger.info(f'Training on {len(texts)} texts')

        # documents = [TaggedDocument(gensim.utils.simple_preprocess(text), [idx]) for idx, text in enumerate(texts)]
        documents = [TaggedDocument(gensim.utils.simple_preprocess(text), [self.idx2doc_id[idx]]) for idx, text in enumerate(texts)]

        logger.info(f'Training on {len(documents)} documents')

        model = gensim.models.doc2vec.Doc2Vec(documents, vector_size=self.vector_size, window=self.window, min_count=1,
                                              workers=multiprocessing.cpu_count())

        self.model = model.docvecs

    def save_word2vec_format(self, file_path, override=False, **kwargs):
        if override and os.path.exists(file_path):
            logger.debug(f'Override {file_path}')
            os.remove(file_path)
            
        self.model.save_word2vec_format(file_path, prefix='', **kwargs)


class WeightedAvgWordVectorsRecSys(KeyedVectorRecSys):
    """
    w2v_model = KeyedVectors.load_word2vec_format('/home/mostendorff/datasets/word_embeddings/glove.6B.200d.word2vec_format.txt', binary=False)

    """
    stop_words = 'english'
    count_vector_size = 100000
    w2v_model = None  # type: KeyedVectors

    def train(self, texts: List):
        if not self.w2v_model:
            raise ValueError('Underlying word2vec model, e.g. GloVe, needs to be set!')

        # reset
        self.model = KeyedVectors(vector_size=self.w2v_model.vector_size)
        self.vector_size = self.w2v_model.vector_size

        count_vec = CountVectorizer(stop_words=self.stop_words, analyzer='word', lowercase=True,
                                    ngram_range=(1, 1), max_features=self.count_vector_size)

        # Transforms the data into a bag of words
        count_train = count_vec.fit(texts)
        idx2bow = count_vec.transform(texts)
        vidx2word = {v: k for k, v in count_train.vocabulary_.items()}

        assert len(vidx2word) == len(count_train.vocabulary_)

        print(f'Vocab size: {len(count_train.vocabulary_)}')

        iterator = self.get_progress_iterator(texts, total=len(texts))

        for idx, text in enumerate(iterator):

            bow = idx2bow[idx].A[0]
            # bow.shape
            #
            # print(bow)

            # words_count = bow.sum()  # no need to total count when using np.average
            vectors = []
            weights = []

            for _idx, count in enumerate(bow):
                if count > 0:
                    word = vidx2word[_idx]
                    try:
                        v = self.w2v_model.get_vector(word)
                        vectors.append(v)
                        weights.append(count)
                    except KeyError:
                        pass

                    pass

            # Check if at least one document term exists as word vector
            if vectors and weights:
                # Weight avg
                doc = np.average(np.array(vectors), axis=0, weights=np.array(weights))

                # Add to model with doc_id
                self.model.add([str(self.idx2doc_id[idx])], [doc])
            else:
                logger.debug(f'Cannot add document {self.idx2doc_id[idx]} due to missing word vectors')

        return self.model
