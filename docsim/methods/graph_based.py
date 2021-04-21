import logging
from collections import defaultdict
from importlib import import_module
from typing import List, Tuple

import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from data_loader import get_pair_id
from docsim.methods import RecSys

stop_words = set(stopwords.words('english'))


logger = logging.getLogger(__name__)


class GraphEmbeddingRecSys(RecSys):
    graph_model_cls = None  # type: str
    graph_model_kwargs = dict()
    graph_model_fit_kwargs = dict()
    doc_id2sub_graph_idx = {}
    sub_graphs = []
    # gold_standard = None  # type: GoldStandard
    include_seeds = None  # type: set
    vector_size = None

    min_nodes_per_sub_graph = 2
    node_embedding_slice = None

    def get_graph_model(self, class_name=None):
        if not class_name:
            class_name = self.graph_model_cls

        if not isinstance(class_name, str):
            raise ValueError('graph model class must be given as package path, e.g., karateclub.DeepWalk')

        # dynamic import for classes
        logger.info(f'Graph algorithm: {class_name}')

        package, module_name = class_name.rsplit('.', 1)
        module = import_module(package)
        cls = getattr(module, module_name)

        return cls(**self.graph_model_kwargs)

    def before_save_to_disk(self):
        # remove stuff before pickle
        if self.include_seeds:
            del self.include_seeds

    def get_node_embedding(self, embedding_matrix, node_idx):
        if self.node_embedding_slice:
            return embedding_matrix[node_idx][self.node_embedding_slice]
        else:
            return embedding_matrix[node_idx]

    def train(self, cits: List):
        import networkx as nx

        # Init full graph
        g = nx.Graph()
        g.add_edges_from(cits)

        # Generate a sorted list of connected components, largest first.
        connected_sub_graphs = sorted(nx.connected_components(g), key=len, reverse=True)

        # We must generate separate embeddings for each graph component (sub-graph)
        self.doc_id2sub_graph_idx = {}
        self.sub_graphs = []

        logger.info(f'Connected sub-graphs: {len(connected_sub_graphs)}')

        for idx, sub_graph_nodes in enumerate(connected_sub_graphs):
            if len(sub_graph_nodes) <= self.min_nodes_per_sub_graph:
                # Skip sub graphs with too few nodes
                continue

            # node ids to numeric indexes
            node_idx2doc_id = [doc_id for node_idx, doc_id in enumerate(sub_graph_nodes)]
            doc_id2node_idx = {doc_id: node_idx for node_idx, doc_id in enumerate(sub_graph_nodes)}
            sg_edges = [(doc_id2node_idx[_from], doc_id2node_idx[_to]) for _from, _to in cits if
                        _from in doc_id2node_idx and _to in doc_id2node_idx]

            # init new graph
            sg = nx.Graph()
            sg.add_edges_from(sg_edges)

            # sub graph must be connected for graph embedding training
            if not nx.is_connected(sg):
                raise ValueError(f'Sub-graph #{idx} is not connected!')

            logger.info(f'Sub-graph #{idx}: {len(sg.nodes)} nodes, {len(sg.edges)} edges')

            graph_embedding = self.get_graph_embeddings(sg)

            # auto set vector size if no custom slicing is used
            if not self.node_embedding_slice:
                self.vector_size = graph_embedding.shape[1]
            elif self.vector_size is None:
                raise ValueError('You must set `vector_size` when using custom `node_embedding_slice`.')

            # convert matrix into keyed vector
            kv = KeyedVectors(vector_size=self.vector_size)

            logger.info(f'Graph embeddings trained: {graph_embedding.shape}')

            # extract embeddings from matrix into keyed vector
            for node_idx in range(len(graph_embedding)):
                doc_vec = self.get_node_embedding(graph_embedding, node_idx)
                doc_id = node_idx2doc_id[node_idx]

                # We need only those docs in include_seeds
                if not self.include_seeds or doc_id in self.include_seeds:
                    kv.add(doc_id, doc_vec)

            logger.info(f'Keyed vectors: {len(kv.vocab)}')

            # save sub_graph
            self.sub_graphs.append(dict(
                # g=sg,
                keyed_vectors=kv,
                node_idx2doc_id=node_idx2doc_id,
                doc_id2node_idx=doc_id2node_idx,
            ))

            # remember sub_group memberships
            for doc_id in sub_graph_nodes:
                self.doc_id2sub_graph_idx[doc_id] = idx

        logger.info('Training completed')

    def get_graph_embeddings(self, sub_graph):

        if self.graph_model_cls.startswith('karateclub.'):

            # init embedding model
            graph_model = self.get_graph_model()
            graph_model.fit(sub_graph)

            return graph_model.get_embedding()
        elif self.graph_model_cls == 'node2vec.Node2Vec':
            self.graph_model_kwargs['graph'] = sub_graph

            graph_model = self.get_graph_model()
            out = graph_model.fit(**self.graph_model_fit_kwargs)

            return out.wv.syn0

        else:
            raise ValueError(f'Graph model is not supported: {self.graph_model_cls}')

    def retrieve_recommendations(self, seed_doc_id: str):
        """
        Find sub-graph first and then do nearest neighbour search in sub-graph.

        :param seed_doc_id: Document id as string
        :return: List[str] List of document ids
        """
        try:
            return [rec_doc_id for rec_doc_id, _ in self.sub_graphs[self.doc_id2sub_graph_idx[seed_doc_id]]['keyed_vectors'].most_similar(seed_doc_id, topn=self.top_k)]
        except KeyError:
            logger.warning(f'Cannot retrieve recommendations, seed ID does not exist:{seed_doc_id}')
            return []


class BibliographicCouplingRecSys(RecSys):
    cits_by_source = {}

    def train(self, cits_by_source: dict):
        # no actual training is performed
        self.cits_by_source = cits_by_source

    def retrieve_recommendations(self, seed_doc_id: str) -> List[str]:
        if seed_doc_id not in self.doc_id2idx:
            raise ValueError(f'Seed ID not found in index: {seed_doc_id}')

        if seed_doc_id not in self.cits_by_source:
            # Seed has no bibliography
            return []

        seed_cits = set(self.cits_by_source[seed_doc_id])

        candidates = []  # (oid, score), ...

        for idx in self.idx2doc_id:
            candidate_oid = self.idx2doc_id[idx]

            # seed cannot be candidate, candidate
            if candidate_oid != seed_doc_id and candidate_oid in self.cits_by_source:
                candidate_cits = set(self.cits_by_source[candidate_oid])

                bib_overlap = len(seed_cits & candidate_cits)

                if bib_overlap > 0:
                    candidates.append((candidate_oid, bib_overlap))

        candidates.sort(key=lambda tup: tup[1], reverse=True)

        return [oid for oid, score in candidates[:self.top_k]]


class CPARecSys(RecSys):
    alpha = -0.85

    def retrieve_recommendations(self, seed_doc_id: str) -> Tuple[List[str], List[str]]:
        """

        Computes CPIs on the fly

        :param seed_doc_id:
        :return:
        """
        # seed_cited_by = set(self.cits_by_target[seed_doc_id])

        if seed_doc_id not in self.cits_by_target:
            logger.warning(f'Opinion is not cited by any other doc: {seed_doc_id}')
            return [], []

        cites_seed = set(self.cits_by_target[seed_doc_id])  # docs that cite seed

        cpa_candidates = []  # (oid, score), ...
        cocit_candidates = []  # (oid, score), ...

        doc_counter = 0
        texts_not_found = 0
        zero_dist = 0

        # Iterate over all candidates (all docs in corpus)
        for candidate_oid in self.cits_by_target:
            cites_candidate = set(self.cits_by_target[candidate_oid])  # docs that cite candidate

            cocitations = cites_candidate & cites_seed
            cocit_count = len(cocitations)

            if cocit_count > 0:
                # CPA
                cpi = 0.

                for citing_doc in cocitations:
                    doc_counter += 1

                    a_pos = self.cits[(citing_doc, seed_doc_id)]  # Position of citation markers
                    b_pos = self.cits[(citing_doc, candidate_oid)]
                    if a_pos and b_pos:
                        # Find smallest distance between citation markers
                        d = max(1, self.get_smallest_distance(a_pos, b_pos))

                        # Distance is relative to document length
                        if citing_doc in self.oid2text_length:
                            tl = self.oid2text_length[citing_doc]

                            if tl > 0:
                                rel_dist = d / self.oid2text_length[citing_doc]

                                # Compute CPI
                                cpi += np.power(rel_dist, self.alpha)  # math.pow(rel_dist, -0.95)
                        else:
                            texts_not_found += 1

                            #    pass
                            # print(f'Text is missing for opinion id = {citing_doc}')

                cpa_candidates.append((candidate_oid, cpi))
                cocit_candidates.append((candidate_oid, cocit_count))

        logger.info(f'doc_counter = {doc_counter}; texts_not_found={texts_not_found}; zero_dist={zero_dist}')  # 17964

        cocit_candidates.sort(key=lambda tup: tup[1], reverse=True)
        cpa_candidates.sort(key=lambda tup: tup[1], reverse=True)

        return (
            [oid for oid, score in cocit_candidates[:self.top_k]],
            [oid for oid, score in cpa_candidates[:self.top_k]],
        )

    def train(self, texts: list, cits_by_source: dict, cits_by_target: dict, cits: dict):
        # no actual training is performed
        self.cits_by_source = cits_by_source
        self.cits_by_target = cits_by_target
        self.oid2text_length = {oid: len(texts[self.doc_id2idx[oid]]) for oid in self.doc_id2idx}
        self.cits = cits

    def get_smallest_distance(self, a_positions, b_positions):
        smalltest_dist = None
        for a in a_positions:
            for b in b_positions:
                dist = abs(a - b)

                if smalltest_dist is None or dist < smalltest_dist:
                    smalltest_dist = dist
                    #return dist
        return smalltest_dist
    #    raise ValueError(a_positions, b_positions)


class TrainedCPARecSys(CPARecSys):
    pair2dist_list = {}
    oid2cocits = defaultdict(set)

    filtered_oids = None


    def train(self, texts: list, cits_by_source: dict, cits_by_target: dict, cits: dict):
        self.cits_by_source = cits_by_source
        self.cits_by_target = cits_by_target
        #self.oid2text_length = {oid: len(texts[self.doc_id2idx[oid]]) for oid in self.doc_id2idx}
        self.cits = cits
        self.pair2dist_list = {}  # (a, b) => list(cpi_1, cpi_2, ...)   # len(list(...)) == co cit
        self.oid2cocits = defaultdict(set)

        # As in Citolytics
        iterator = self.get_progress_iterator(self.idx2doc_id.items(), total=len(self.idx2doc_id))

        for idx, citing_doc in iterator:
            # Link pairs in document oid
            if citing_doc in cits_by_source:
                processed_pairs = set()

                for a_oid in cits_by_source[citing_doc]:
                    for b_oid in cits_by_source[citing_doc]:
                        if a_oid != b_oid:
                            if self.filtered_oids is not None and a_oid not in self.filtered_oids and b_oid not in self.filtered_oids:
                                # Skip if candidate is not in target corpus
                                continue

                            pair_id = get_pair_id(a_oid, b_oid)

                            if pair_id not in processed_pairs:

                                a_pos = cits[(citing_doc, a_oid)]
                                b_pos = cits[(citing_doc, b_oid)]

                                dist = self.get_smallest_distance(a_pos, b_pos)
                                d = (citing_doc, dist)

                                if pair_id in self.pair2dist_list:
                                    self.pair2dist_list[pair_id].append(d)
                                else:
                                    self.pair2dist_list[pair_id] = [d]

                                    # Save cocitations to speed up retrieval
                                    self.oid2cocits[a_oid].add(b_oid)
                                    self.oid2cocits[b_oid].add(a_oid)

                                processed_pairs.add(pair_id)

    def retrieve_recommendations(self, seed_doc_id: int) -> List[int]:

        cocits = self.oid2cocits[seed_doc_id]

        cpa_candidates = []

        for candidate_oid in cocits:
            if self.filtered_oids is not None and candidate_oid not in self.filtered_oids:
                # Skip if candidate is not in target corpus
                continue

            pair_id = get_pair_id(seed_doc_id, candidate_oid)
            cpi = 0.
            for citing_doc, dist in self.pair2dist_list[pair_id]:

                # Compute CPI
                cpi += np.power(dist, self.alpha)

            cpa_candidates.append((candidate_oid, cpi))

        cpa_candidates.sort(key=lambda tup: tup[1], reverse=True)

        return [oid for oid, score in cpa_candidates[:self.top_k]]


