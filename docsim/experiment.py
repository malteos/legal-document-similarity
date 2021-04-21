import json
import logging
import os
import pickle
from pathlib import Path

from gensim.models import KeyedVectors
from smart_open import open

from docsim.gold_standard import GoldStandard
from docsim.methods import RecSys
from docsim.methods.keyed_vector_based import KeyedVectorRecSys

logger = logging.getLogger(__name__)


class Experiment(object):
    exp_dir = None
    models_dir = None
    gs = None
    systems = None
    
    w2v_paths = {}
    w2v_models = {}

    # docs
    # docs = None
    doc_id2idx = None
    idx2doc_id = None
    texts = None
    cits = None

    limited_texts = {}
    
    results_df = None

    def __init__(self, name, env, data_dir: Path, pretty_name=None):
        self.name = name
        self.env = env
        self.data_dir = data_dir
        
        if pretty_name:
            self.pretty_name = pretty_name
        else:
            self.pretty_name = name

    def load_data(self):
        """

        Required input files in ./data:

        ./ocb
        - texts.json.gz
        - cits.json
        - doc_id2idx.json
        - idx2doc_id.json
        - gold.csv

        ./wikisource
        - cl_dump.pickle.gz
        - gold.csv

        Word vectors:
        - /glove.6B/glove.6B.300d.w2vformat.txt
        - ocb_and_wikisource.w2v.txt.gz
        - /fasttext/wiki-news-300d-1M.vec
        - ocb_and_wikisource.fasttext.w2v.txt.gz

        tar -cvzf wikisource.tar.gz docs_without_text.json doc_id2idx.json idx2doc_id.json texts.json.gz gold.csv meta.csv
        tar -cvzf ocb.tar.gz cits.json doc_id2idx.json idx2doc_id.json texts.json.gz gold.csv meta.csv

        """

        if self.name == 'ocb':
            self.exp_dir = self.data_dir / 'ocb'
            self.models_dir = Path('./models/ocb')

            # Load everything from disk
            # meta_df = pd.read_csv(self.exp_dir / 'meta.csv', dtype={'id': 'str'}).set_index('id')

            self.texts = json.load(open(self.exp_dir / 'texts.json.gz', 'r'))
            self.cits = json.load(open(self.exp_dir / 'cits.json', 'r'))

            self.doc_id2idx = json.load(open(self.exp_dir / 'doc_id2idx.json', 'r'))
            self.idx2doc_id = json.load(open(self.exp_dir / 'idx2doc_id.json', 'r'))

            logger.info(f'Documents loaded: {len(self.doc_id2idx):,}')

            self.gs = GoldStandard(self.exp_dir / 'gold.csv')

            logger.info(f'Unique documents in gold standard: {len(self.gs.doc_ids):,}')

        elif self.name == 'wikisource':
            self.exp_dir = self.data_dir / 'wikisource'
            self.models_dir = Path('./models/wikisource')

            docs = json.load(open(self.exp_dir / 'docs_without_text.json', 'r'))
            doc_id2idx = json.load(open(self.exp_dir / 'doc_id2idx.json', 'r'))
            idx2doc_id = json.load(open(self.exp_dir / 'idx2doc_id.json', 'r'))

            self.texts = json.load(open(self.exp_dir / 'texts.json.gz', 'r'))

            # with open(self.exp_dir / 'cl_dump.pickle.gz', 'rb') as f:
            #     docs, idx2doc_id, doc_id2idx, self.texts, _, _, _ = pickle.load(f)

            # Convert doc_ids to str
            self.idx2doc_id = {idx: str(doc_id) for idx, doc_id in idx2doc_id.items()}
            self.doc_id2idx = {str(doc_id): idx for doc_id, idx in doc_id2idx.items()}

            self.gs = GoldStandard(self.exp_dir / 'gold.csv')

            # Citations
            def get_cits_from_docs(doc_index, gs):
                for idx, d in doc_index.items():
                    citing_id = str(d['id'])
                    # citing_id = d['id']

                    if citing_id not in gs.doc_ids:
                        # doc must part of gold standard
                        # continue
                        pass

                    for cit_url in d['opinions_cited']:
                        # extract id from api url
                        cited_id = cit_url[54:-1]

                        if cited_id.isdigit():
                            # cited_id = int(cited_id)
                            yield (citing_id, cited_id)

            self.cits = list(get_cits_from_docs(docs, self.gs))

        else:
            raise ValueError(f'Invalid experiment select: {self.name}')

        ###
        self.w2v_paths = {
            'glove': self.env['datasets_dir'] + '/glove.6B/glove.6B.300d.w2vformat.txt',
            'glove_custom': self.data_dir / 'ocb_and_wikisource.w2v.txt.gz',
            'fasttext': self.env['datasets_dir'] + '/fasttext/wiki-news-300d-1M.vec',
            'fasttext_custom': self.data_dir / 'ocb_and_wikisource.fasttext.w2v.txt.gz',
        }
        
        # Create directories
        if not os.path.exists(self.models_dir):
            logger.info(f'Creating new models dir: {self.models_dir}')
            os.makedirs(self.models_dir)

    def filter_docs(self):
        # Citations
        doc_ids_with_cits = set({f for f, t in self.cits}).union({t for f, t in self.cits})

        # Filter
        filtered_texts = []
        filtered_idx2doc_id = {}
        filtered_doc_id2idx = {}

        for idx, text in enumerate(self.texts):
            doc_id = self.idx2doc_id[idx]

            if doc_id in self.gs.doc_ids and len(text) > 1 and doc_id in doc_ids_with_cits:
                filtered_idx2doc_id[len(filtered_texts)] = doc_id
                filtered_doc_id2idx[doc_id] = len(filtered_texts)
                filtered_texts.append(text)

        logger.info(f'Documents after filtering: {len(filtered_texts):,} (before {len(self.texts):,})')

        # Override existing data
        del self.texts
        del self.idx2doc_id
        del self.doc_id2idx

        self.texts = filtered_texts
        self.idx2doc_id = filtered_idx2doc_id
        self.doc_id2idx = filtered_doc_id2idx

    def get_common_kwargs(self):
        return dict(
            doc_id2idx=self.doc_id2idx,
            idx2doc_id=self.idx2doc_id,
            print_progress=True,
        )

    def get_limited_texts(self, max_length: int):
        if max_length not in self.limited_texts:
            self.limited_texts[max_length] = [' '.join(t.split()[:max_length]) for t in self.texts]

        return self.limited_texts[max_length]

    def get_w2v_model(self, name):
        if name not in self.w2v_paths:
            raise ValueError(f'No w2v model exists for: {name}')

        if name not in self.w2v_models:
            self.w2v_models[name] = KeyedVectors.load_word2vec_format(self.w2v_paths[name])

        return self.w2v_models[name]

    def get_included_seeds(self):
        # Seed Doc IDs that are used in experiment (for evaluation)
        return set(self.doc_id2idx.keys())

    def get_systems(self, common_kwargs, models_dir=None):
        if not self.systems:
            if not models_dir:
                models_dir = self.models_dir

            systems = {}

            for fn in os.listdir(models_dir):  # type: str
                fp = models_dir / fn

                if os.path.isfile(fp):
                    # Pickle's
                    if fn.endswith('.pickle'):
                        rs = RecSys.load_from_disk(fp, **common_kwargs)
                        rs_name = fn.replace('.pickle', '')

                    # Auto-load w2v's from models dir
                    elif fn.endswith('.w2v.txt'):  # and 'bert' in fn:
                        rs = KeyedVectorRecSys(**common_kwargs)
                        rs.load_word2vec_format(fp)

                        rs_name = fn.replace('.w2v.txt', '')
                    else:
                        logger.warning(f'Cannot load model from: {fp}')
                        continue

                    # Replace with pretty name
                    for n, r in self.get_pretty_systems().items():
                        rs_name = rs_name.replace(n, r)

                    if rs_name in systems:
                        raise ValueError(f'System name exists already: {rs_name}')

                    systems[rs_name] = rs

            logger.info(f'Loaded systems: {systems.keys()}')
            
            self.systems = systems
            
        return self.systems

    @staticmethod
    def get_pretty_systems():
        return {
            'tfidf': 'TF-IDF',
            'deepwalk': 'DeepWalk',
            'boostne': 'BoostNE',
            'node2vec': 'Node2Vec',
            'walklets': 'Walklets',
            'doc2vec': 'Paragraph Vectors',
            'poincare': 'Poincaré',
            'roberta-': 'RoBERTa ',
            'bert-': 'BERT ',
            'longformer-': 'Longformer ',
            'avg_glove': 'AvgGloVe',
            'avg_fasttext': 'AvgFastText',
            '_mean': '-mean',
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Experiment(name={self.name})'
