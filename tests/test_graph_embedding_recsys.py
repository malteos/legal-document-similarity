import logging
import random

import networkx as nx
from gensim.models.poincare import PoincareModel

from docsim.gold_standard import GoldStandard
from docsim.methods.graph_based import GraphEmbeddingRecSys
from tests import OUT_DIR, BaseRecSysTest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(0)


class GraphEmbeddingRecSysTest(BaseRecSysTest):
    cits = None
    common_kwargs = None

    def get_citation_graph(self, doc_count=100, removed_edges=75, seed=0):
        # Generate random graph
        g = nx.newman_watts_strogatz_graph(doc_count, 5, 0.01, seed=seed)

        # Remove random edges
        for a, b in random.sample(g.edges, k=removed_edges):
            g.remove_edge(a, b)

        self.assertFalse(nx.is_connected(g), 'Random generated graph must be NOT connected')

        # Convert to citations (int => str)
        cits = [(str(a), str(b)) for a,b in g.edges]

        return cits, g

    def test_deep_walk_train_save_load(self):
        out_fp = OUT_DIR / 'deep_walk'

        doc_count = 100
        cits, _ = self.get_citation_graph(doc_count)

        doc_id2idx = {str(i): i for i in range(doc_count)}
        idx2doc_id = [str(i) for i in range(doc_count)]

        self.common_kwargs = dict(doc_id2idx=doc_id2idx, idx2doc_id=idx2doc_id)

        gs = GoldStandard()
        gs._doc_ids = set(idx2doc_id)

        rs = GraphEmbeddingRecSys(**self.common_kwargs,
                                  gold_standard=gs,
                                  graph_model_cls='karateclub.DeepWalk',
                                  graph_model_kwargs=dict(dimensions=100)
                                  )
        rs.train(cits)
        rs.save_to_disk(out_fp)

        seed_id = '0'

        self.assertEqual(10, len(rs.retrieve_recommendations(seed_id)))
        self.assertEqual(100, rs.vector_size)

        # Load again
        rs2 = GraphEmbeddingRecSys.load_from_disk(out_fp, **self.common_kwargs)

        self.assertEqual(
            rs.retrieve_recommendations(seed_id),
            rs2.retrieve_recommendations(seed_id),
            'Loaded model does not yield same results as fresh model'
        )

    def test_poincare(self):
        doc_count = 100
        cits, _ = self.get_citation_graph(doc_count)

        poincare_model = PoincareModel(
            cits,
            size=300,
            alpha=0.1,
            negative=10,
            workers=1,
            epsilon=1e-05,
            regularization_coeff=1.0,
            burn_in=10,
            burn_in_alpha=0.01,
            init_range=(-0.001, 0.001),
        )
        poincare_model.train(
            epochs=2,
        )

        print(poincare_model.kv.vector_size)

    def test_boostne(self):
        doc_count = 100
        cits, _ = self.get_citation_graph(doc_count)

        doc_id2idx = {str(i): i for i in range(doc_count)}
        idx2doc_id = [str(i) for i in range(doc_count)]

        self.common_kwargs = dict(doc_id2idx=doc_id2idx, idx2doc_id=idx2doc_id)

        gs = GoldStandard()
        gs._doc_ids = set(idx2doc_id)

        dim = 20
        iters = 14
        output_size = dim * (iters + 1)

        boostne = GraphEmbeddingRecSys(
            include_seeds=gs.doc_ids,
            #vector_size=dim,
            graph_model_cls='karateclub.BoostNE',
            graph_model_kwargs=dict(
                # dimension of each single level embedding
                dimensions=dim,  # 8
                order=2,  # 2
                # number of levels of the final multi-level embedding (k=8, as in the paper; or k=16 results for CORA dataset in paper)
                iterations=iters, # 16
                alpha=0.01,
            ),
            #node_embedding_slice=slice(dim * iters, dim * (iters + 1)),
            **self.common_kwargs
        )
        boostne.train(cits)

        # vector size = dim + dim * iters

        seed_id = '0'
        vector_size = boostne.sub_graphs[boostne.doc_id2sub_graph_idx[seed_id]]['keyed_vectors'].vector_size

        # self.assertEqual(vector_size, dim)
        self.assertEqual(vector_size, 300)
        # self.assertEqual(vector_size, dim + dim * iters)
        print(vector_size)


    def test_walklets(self):
        doc_count = 100
        cits, _ = self.get_citation_graph(doc_count)

        doc_id2idx = {str(i): i for i in range(doc_count)}
        idx2doc_id = [str(i) for i in range(doc_count)]

        self.common_kwargs = dict(doc_id2idx=doc_id2idx, idx2doc_id=idx2doc_id)

        gs = GoldStandard()
        gs._doc_ids = set(idx2doc_id)

        ger = GraphEmbeddingRecSys(
            include_seeds=gs.doc_ids,
            #vector_size=dim,
            graph_model_cls='karateclub.Walklets',
            graph_model_kwargs=dict(
                dimensions=60,#100,  # Number of embedding dimension. Default is 32.
                window_size=5, #3,
                # iteration=10, #  Number of SVD iterations. Default is 10.
                # order=2,  # Number of PMI matrix powers. Default is 2.
            ),
            #node_embedding_slice=slice(dim * iters, dim * (iters + 1)),
            **self.common_kwargs
        )
        ger.train(cits)

        # vector size = dim + dim * iters

        seed_id = '0'
        vector_size = ger.sub_graphs[ger.doc_id2sub_graph_idx[seed_id]]['keyed_vectors'].vector_size

        # self.assertEqual(vector_size, dim)
        self.assertEqual(vector_size, 300)
        # self.assertEqual(vector_size, dim + dim * iters)
        print(vector_size)


    def test_node2vec(self):
        doc_count = 100
        cits, _ = self.get_citation_graph(doc_count)

        doc_id2idx = {str(i): i for i in range(doc_count)}
        idx2doc_id = [str(i) for i in range(doc_count)]

        self.common_kwargs = dict(doc_id2idx=doc_id2idx, idx2doc_id=idx2doc_id)

        gs = GoldStandard()
        gs._doc_ids = set(idx2doc_id)

        rs = GraphEmbeddingRecSys(
            include_seeds=gs.doc_ids,
            # vector_size=dim,
            graph_model_cls='node2vec.Node2Vec',
            graph_model_kwargs=dict(
                dimensions=100, walk_length=5, num_walks=10,
                workers=4
            ),
            graph_model_fit_kwargs=dict(
                window=5, min_count=1, batch_words=4
            ),
            # node_embedding_slice=slice(dim * iters, dim * (iters + 1)),
            **self.common_kwargs
        )
        rs.train(cits)

        seed_id = '0'

        self.assertEqual(10, len(rs.retrieve_recommendations(seed_id)))
        self.assertEqual(100, rs.vector_size)
