import json
import logging
import os
from collections import defaultdict
from typing import Dict

import pandas as pd
import tqdm
from pandas import DataFrame

from docsim.methods import RecSys
from utils import get_avg_precision, get_reciprocal_rank

logger = logging.getLogger(__name__)


class GoldStandard(object):
    seed_col = 'seed_id'
    target_col = 'target_id'
    label_col = 'label'
    seed_ids_without_recommendations = []
    
    _doc_ids = set()
    _results = []

    ret_docs_by_system_and_seed = None
    

    """
    df = pd.DataFrame(rows, columns=['seed_id', 'target_id', 'label'])
    df.head()
    """
    def __init__(self, file_path=None):
        if file_path:
            self.df = pd.read_csv(file_path, dtype={
                self.seed_col: 'str', 
                self.target_col: 'str', 
                #self.label_col: 'int'
            })

    @property
    def doc_ids(self):
        if not self._doc_ids:
            self._doc_ids = set(self.df[self.seed_col].values.tolist() + self.df[self.target_col].values.tolist())

        return self._doc_ids

    def evaluate(self, systems: Dict[str, RecSys], include_seeds: set=None, show_warning=False, show_progress=False, tqdm_notebook=False, cache_dir: str = None) -> DataFrame:
        """

        systems = {
            'Doc2Vec': doc2vec,
            'AvgGloVe': avg_glove,
            'BERT': bert,
            #'DeepWalk': deepwalk,
            'Poincare': poincare,
        }

        :param cache_dir: Store retrieved recommendations in this directory for each system
        :param include_seeds: Seed ids must be part of this set
        :param systems:
        :return: DataFrame
        """

        df = self.df

        if include_seeds:
            logger.info(f'Include only {len(include_seeds)} seeds for evaluation')
            df = df[df[self.seed_col].isin(include_seeds)]

        # Group by seed id
        grouped_by = df.groupby(self.seed_col)

        self.seed_ids_without_recommendations = []
        self._results = []

        if show_progress:
            if tqdm_notebook:
                grouped_by = tqdm.tqdm_notebook(grouped_by, total=len(grouped_by))
            else:
                grouped_by = tqdm.tqdm(grouped_by, total=len(grouped_by))

        # Load cache
        cache = {}

        if cache_dir:
            if os.path.exists(cache_dir):
                # Cache for each system
                for name, recsys in systems.items():
                    fp = os.path.join(cache_dir, f'{name}.json')
                    if os.path.exists(fp):
                        cache[name] = json.load(open(fp))
                    else:
                        cache[name] = {}

            else:
                logger.info(f'Creating new cache dir: {cache_dir}')
                os.makedirs(cache_dir)

        # Track coverage
        self.ret_docs_by_system_and_seed = {n: defaultdict(list) for n in systems}

        for seed_id, gp in grouped_by:
            # Evaluate for each seed
            rel_docs = gp[self.target_col].values.tolist()

            result = {
                self.seed_col: seed_id,
                'rel_docs_count': len(rel_docs),
            }

            # Get results for each system
            for name, recsys in systems.items():
                try:
                    # Load from cache if possible
                    if name in cache and seed_id in cache[name]:
                        ret_docs = cache[name][seed_id]
                    else:
                        # no cache
                        ret_docs = recsys.retrieve_recommendations(seed_id)

                        if name not in cache:
                            cache[name] = {}
                            
                        cache[name][seed_id] = ret_docs

                    rel_ret_docs_count = len(set(ret_docs) & set(rel_docs))

                    self.ret_docs_by_system_and_seed[name][seed_id] = ret_docs  # Track coverage

                    result[f'{name}_ret'] = len(ret_docs)
                    result[f'{name}_rel'] = rel_ret_docs_count

                    if ret_docs and rel_docs:
                        # Precision = No. of relevant documents retrieved / No. of total documents retrieved
                        result[f'{name}_p'] = rel_ret_docs_count / len(ret_docs)

                        # Recall = No. of relevant documents retrieved / No. of total relevant documents
                        result[f'{name}_r'] = rel_ret_docs_count / len(rel_docs)

                        # Avg. precision (for MAP)
                        result[f'{name}_avg_p'] = get_avg_precision(ret_docs, rel_docs)

                        # Reciprocal rank (for MRR)
                        result[f'{name}_reciprocal_rank'] = get_reciprocal_rank(ret_docs, rel_docs)

                except (IndexError, ValueError) as e:                    
                    self.seed_ids_without_recommendations.append(seed_id)
                    
                    if show_warning:
                        logger.warning(f'Cannot retrieve recommendations for #{seed_id} from {name}: {e}')
                    
                    result[f'{name}_ret'] = 0
                    result[f'{name}_rel'] = 0
                    result[f'{name}_p'] = 0
                    result[f'{name}_r'] = 0
                    result[f'{name}_avg_p'] = 0
                    result[f'{name}_reciprocal_rank'] = 0

            self._results.append(result)

        # Save cache
        if cache_dir:
            for name, recsys in systems.items():
                if name in cache:
                    fp = os.path.join(cache_dir, f'{name}.json')
                    json.dump(cache[name], open(fp, 'w'))

                    logger.info(f'Saving cache for "{name}" in {fp}')

        # Build data frame
        df = pd.DataFrame(self._results).set_index('seed_id')

        return df


class LegalWikiSource(GoldStandard):
    pass

