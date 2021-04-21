import logging
import gensim
import json
import os
import requests
import pickle
import pandas as pd
import logging
from pathlib import Path
from docsim.gold_standard import GoldStandard
from tqdm import tqdm_notebook as tqdm
from smart_open import open
from docsim.environment import get_env

logger = logging.getLogger(__name__)


def extract_text(data_dir='./data', workers=None):
    """

    Extract plain-text from corpora as input to word2vec

    data/wikisource/cl_dump.pickle.gz
    data/ocb/texts.json.gz

    Output: ocb_and_wikisource.w2v_tokens.txt

    :param data_dir: Path to input/output directory
    :param workers:
    :return:
    """
    # TODO for word2vec

    env = get_env()
    data_dir = Path(data_dir)

    wikisource_data_dir = data_dir / 'wikisource'
    ocb_data_dir = data_dir / 'ocb'

    if not workers:
        workers = env['workers']

    logger.info(f'Using {workers} workers')

    # Load everything from disk
    ocb_texts = json.load(open(ocb_data_dir / 'texts.json.gz', 'r'))

    with open(wikisource_data_dir / 'cl_dump.pickle.gz', 'rb') as f:
        _, _, _, wikisource_texts, _, _, _ = pickle.load(f)

    # Tokenized text for fastText + GloVe
    tokens_count = 0
    with open(data_dir / 'ocb_and_wikisource.w2v_tokens.txt', 'w') as f:
        for text in ocb_texts:
            for token in gensim.utils.simple_preprocess(text, min_len=2, max_len=15):
                f.write(token + ' ')
                tokens_count += 1
            f.write('\n')
        for text in wikisource_texts:
            for token in gensim.utils.simple_preprocess(text, min_len=2, max_len=15):
                f.write(token + ' ')
                tokens_count += 1
            f.write('\n')

    logger.info(f'Total tokens: {tokens_count:,}')


def train_fasttext(data_dir='./data', dim=300, epoch=5, ft_model='skipgram', ft_lr=0.05, ft_window=5):

    data_dir = Path(data_dir)

    import fasttext

    model = fasttext.train_unsupervised(str(data_dir / 'ocb_and_wikisource.w2v_tokens.txt'),
                                        model=ft_model,
                                        lr=ft_lr,  # learning rate [0.05]
                                        dim=dim,  # size of word vectors [100]
                                        ws=ft_window,  # size of the context window [5]
                                        epoch=epoch  # number of epochs [5]
                                        # thread            # number of threads [number of cpus]
                                        )
    model.save_model(str(data_dir / 'ocb_and_wikisource.fasttext.bin'))

    from gensim.models.wrappers import FastText

    ft_model = FastText.load_fasttext_format(str(data_dir / 'ocb_and_wikisource.fasttext.bin'))

    ft_model.wv.save_word2vec_format(data_dir / 'ocb_and_wikisource.fasttext.w2v.txt')

    logger.info('done')
