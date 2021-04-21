import logging
import os
import shutil
from pathlib import Path
from unittest import TestCase

from docsim.environment import get_env

TEST_DIR = Path(__file__).parent / 'fixtures'
OUT_DIR = TEST_DIR / 'out'


logger = logging.getLogger(__name__)


class BaseRecSysTest(TestCase):
    common_kwargs = None

    def setUp(self) -> None:
        if os.path.exists(OUT_DIR):
            shutil.rmtree(OUT_DIR)

        os.makedirs(OUT_DIR)


class BaseTextRecSysTest(BaseRecSysTest):
    seed_id = '0'
    texts = None
    env = None

    def setUp(self) -> None:
        # Input
        in_fp = TEST_DIR / '1.txt'

        with open(in_fp, 'r') as f:
            self.texts = f.readlines()
        self.assertEqual(5, len(self.texts))

        doc_id2idx = {str(i): i for i in range(len(self.texts))}
        idx2doc_id = [str(i) for i in range(len(self.texts))]

        self.common_kwargs = dict(doc_id2idx=doc_id2idx, idx2doc_id=idx2doc_id, vector_size=100)

        self.env = get_env()