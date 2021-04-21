import logging

from docsim.methods.transformer_based import SentenceTransformerRecSys, TransformerRecSys
from tests import BaseTextRecSysTest, OUT_DIR

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TransformerRecSysTest(BaseTextRecSysTest):
    def test_simple_sentence_transformers(self):
        model = SentenceTransformer(self.env['datasets_dir'] + '/sentence_transformers/bert-base-nli-mean-tokens')

        sentence_embeddings = model.encode(self.texts)

        for sentence, embedding in zip(self.texts, sentence_embeddings):
            print("Sentence:", sentence)
            print("Embedding:", embedding)
            print("")

    def test_simple_sentence_transformers_from_disk(self):
        model = SentenceTransformer(self.env['datasets_dir'] + '/sentence_transformers/bert-base-nli-mean-tokens')

        # sentence_embeddings = model.encode(self.texts)
        #
        # for sentence, embedding in zip(self.texts, sentence_embeddings):
        #     print("Sentence:", sentence)
        #     print("Embedding:", embedding)
        #     print("")

        self.assertEqual(768, model.get_sentence_embedding_dimension())

    def test_sentence_transformers_train_save_load(self):
        out_fp = OUT_DIR / 'sentence_transformer.w2v.txt'

        rs = SentenceTransformerRecSys(**self.common_kwargs,
                                       model_name_or_path=self.env['datasets_dir']+'/sentence_transformers/bert-base-nli-mean-tokens')

        rs.train(self.texts)
        rs.save_word2vec_format(out_fp)

        rs2 = SentenceTransformerRecSys(**self.common_kwargs)
        rs2.load_word2vec_format(out_fp)

        self.assertEqual(768, rs2.vector_size)
        self.assertEqual(768, rs2.vector_size)

        self.assertEqual(rs.retrieve_recommendations(self.seed_id), rs2.retrieve_recommendations(self.seed_id))

    def test_simple_bert_transformer(self):
        model_path = self.env['bert_dir'] + '/bert-base-cased'

        out_fp = OUT_DIR / 'bert.w2v.txt'
        out2_fp = OUT_DIR / 'bert-mean.w2v.txt'

        rs = TransformerRecSys(model_name_or_path=model_path, pooling_strategy='pooled', **self.common_kwargs)
        rs.train(self.texts)
        rs.save_word2vec_format(out_fp)

        rs2 = TransformerRecSys(model_name_or_path=model_path, pooling_strategy='reduce_mean', **self.common_kwargs)
        rs2.train(self.texts)
        rs2.save_word2vec_format(out2_fp)

        self.assertNotEqual(rs.retrieve_recommendations(self.seed_id), rs2.retrieve_recommendations(self.seed_id))




