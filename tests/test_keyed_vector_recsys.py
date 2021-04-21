import logging

import numpy as np
from gensim.models import KeyedVectors

from docsim.methods.keyed_vector_based import EnsembleKeyedVectorRecSys
from docsim.methods.text_based import Doc2VecRecSys, WeightedAvgWordVectorsRecSys
from tests import TEST_DIR, OUT_DIR, BaseTextRecSysTest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeyedVectorRecSysTest(BaseTextRecSysTest):
    def test_doc2vec_train_save_load(self):
        out_fp = OUT_DIR / 'doc2vec'
        out_w2v_fp = OUT_DIR / 'doc2vec_w2v'

        rs = Doc2VecRecSys(**self.common_kwargs)

        rs.train(self.texts)

        self.assertEqual(5, rs.model.count)
        self.assertEqual(100, rs.vector_size)

        rs.save_to_disk(out_fp)

        # Load again
        rs2 = Doc2VecRecSys.load_from_disk(out_fp, **self.common_kwargs)

        seed_id = '0'

        self.assertEqual(
            rs.retrieve_recommendations(seed_id),
            rs2.retrieve_recommendations(seed_id),
            'Loaded model does not yield same results as fresh model'
        )

        # Save as keyed vector
        rs.save_word2vec_format(out_w2v_fp)

        # Load as keyed vector
        rs3 = Doc2VecRecSys(**self.common_kwargs)
        rs3.load_word2vec_format(out_w2v_fp)

        self.assertEqual(
            rs.retrieve_recommendations(seed_id),
            rs3.retrieve_recommendations(seed_id),
            'Loaded Word2Vec model does not yield same results as fresh model'
        )

        logger.info('Retrieved recs: %s' % list(rs.retrieve_recommendations(seed_id)))

    def test_load_w2v(self):
        in_w2v_fp = TEST_DIR / 'doc2vec_w2v.txt'

        wv = KeyedVectors.load_word2vec_format(in_w2v_fp)

        self.assertEqual(100, wv.vector_size)
        self.assertEqual([str(i) for i in range(5)], wv.index2word)

    def test_glove(self):
        out_fp = OUT_DIR / 'avgglove'

        glove = KeyedVectors.load_word2vec_format(TEST_DIR / 'glove.6B.200d.w2vformat.1k.txt')

        rs = WeightedAvgWordVectorsRecSys(w2v_model=glove, **self.common_kwargs)
        rs.train(self.texts)

        seed_id = '0'
        self.assertEqual(rs.vector_size, 200)

        # One document less!
        self.assertEqual(4, len(rs.model.vocab))
        self.assertEqual(['2', '3', '4'], rs.retrieve_recommendations(seed_id))

        rs.save_to_disk(out_fp)

        # Load again
        rs2 = WeightedAvgWordVectorsRecSys.load_from_disk(out_fp, **self.common_kwargs)

        self.assertEqual(
            rs.retrieve_recommendations(seed_id),
            rs2.retrieve_recommendations(seed_id),
            'Loaded model does not yield same results as fresh model'
        )


    def test_load_multi(self):
        in_w2v_fp = TEST_DIR / 'doc2vec_w2v.txt'

        wv = KeyedVectors.load_word2vec_format(in_w2v_fp)
        glove = KeyedVectors.load_word2vec_format(TEST_DIR / 'glove.6B.200d.w2vformat.1k.txt')

        models = [wv, wv, wv, glove]

        target_vector_size = np.sum([m.vector_size for m in models])

        self.assertEqual(models[0].vector_size*3 + 200, target_vector_size)

        # Build new keyed vector model
        model = KeyedVectors(vector_size=target_vector_size)

        # self.assertEqual([str(i) for i in range(5)], wv.index2word)

        # Iterate over all words (in first model)
        for doc_id in models[0].index2word:
            # print(type(doc_id))
            # Stack vectors from all models
            models_vec = []

            for m in models:
                if doc_id in m.index2word:
                    models_vec.append(m.get_vector(doc_id))
                else:
                    print(f'WARNING: {doc_id} does not exist in {m}')
                    models_vec.append(np.zeros((m.vector_size)))

            vec = np.hstack(models_vec)

            model.add(doc_id, vec)

        self.assertEqual(300 + 200, model.get_vector('0').shape[0])


    def test_load_ensemble(self):
        in_w2v_fp = TEST_DIR / 'doc2vec_w2v.txt'
        in_glove_fp = TEST_DIR / 'glove.6B.200d.w2vformat.1k.txt'

        ensemble = EnsembleKeyedVectorRecSys(**self.common_kwargs)
        ensemble.train([in_w2v_fp,in_glove_fp])

        print(ensemble.retrieve_recommendations('1'))

        #
        # target_vector_size = np.sum([m.vector_size for m in models])
        #
        # self.assertEqual(models[0].vector_size*3 + 200, target_vector_size)
        #
        # # Build new keyed vector model
        # model = KeyedVectors(vector_size=target_vector_size)
        #
        # # self.assertEqual([str(i) for i in range(5)], wv.index2word)
        #
        # # Iterate over all words (in first model)
        # for doc_id in models[0].index2word:
        #     # print(type(doc_id))
        #     # Stack vectors from all models
        #     models_vec = []
        #
        #     for m in models:
        #         if doc_id in m.index2word:
        #             models_vec.append(m.get_vector(doc_id))
        #         else:
        #             print(f'WARNING: {doc_id} does not exist in {m}')
        #             models_vec.append(np.zeros((m.vector_size)))
        #
        #     vec = np.hstack(models_vec)
        #
        #     model.add(doc_id, vec)
        #
        # self.assertEqual(300 + 200, model.get_vector('0').shape[0])
