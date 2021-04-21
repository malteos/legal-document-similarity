import logging

from docsim.methods.text_based import TfIdfRecSys
from tests import BaseTextRecSysTest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextBasedRecSysTest(BaseTextRecSysTest):
    def test_tfidf(self):
        rs = TfIdfRecSys(**self.common_kwargs)
        rs.train(self.texts)
        rs.top_k = 3
        recs = rs.retrieve_recommendations(self.seed_id)

        print(recs)

        self.assertEqual(rs.top_k, len(recs))
        self.assertNotIn(self.seed_id, recs)
