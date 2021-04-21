import logging
import os
from pathlib import Path
from distutils.version import LooseVersion
import fire
import transformers
from gensim.models import KeyedVectors

from docsim.environment import get_env
from docsim.experiment import Experiment
from docsim.methods.graph_based import GraphEmbeddingRecSys
from docsim.methods.text_based import TfIdfRecSys, Doc2VecRecSys, WeightedAvgWordVectorsRecSys
from docsim.methods.transformer_based import TransformerRecSys, SentenceTransformerRecSys

logger = logging.getLogger(__name__)


def compute_doc_vecs(experiment, data_dir='./data', workers=None, override=False, dense_vector_size=300,
                        sparse_vector_size=500000, gpu=None):
    """

    Examples:

    python cli.py compute_doc_vecs wikisource --override=1 --gpu 0
    python cli.py compute_doc_vecs ocb --override=1 --gpu 1


    :param data_dir: Path to data (for input and output)
    :param experiment: Experiment name (ocb or wikisource)
    :param workers: Number of workers
    :param override: Override existing output
    :param dense_vector_size: Size of dense document vectors (avg word2vec, graph embeddings, ...)
    :param sparse_vector_size: Size of sparse document vectors (TF-IDF)
    :param cuda_device: Use CUDA device for Transformer models
    :return:
    """
    env = get_env()
    data_dir = Path(data_dir)

    logger.info(f'Experiment: {experiment}')

    exp = Experiment(name=experiment, env=env, data_dir=data_dir)

    exp.load_data()
    exp.filter_docs()

    models_dir = exp.models_dir
    common_kwargs = exp.get_common_kwargs()

    if not workers:
        workers = env['workers']

    logger.info(f'Using {workers} workers')

    if gpu:
        logger.info(f'Using CUDA device={gpu}')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # TF-IDF
    out_fp = models_dir / 'tfidf.pickle'
    if override or not os.path.exists(out_fp):
        rs = TfIdfRecSys(vector_size=sparse_vector_size, **common_kwargs)
        rs.train(exp.texts)
        rs.save_to_disk(out_fp, override=override)

    # Doc2Vec
    out_fp = models_dir / 'doc2vec.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = Doc2VecRecSys(**common_kwargs, vector_size=dense_vector_size)
        rs.train(exp.texts)
        rs.save_word2vec_format(out_fp, override=override)

    out_fp = models_dir / 'doc2vec_512.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = Doc2VecRecSys(**common_kwargs, vector_size=dense_vector_size)
        rs.train(exp.get_limited_texts(512))
        rs.save_word2vec_format(out_fp, override=override)

    out_fp = models_dir / 'doc2vec_4096.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = Doc2VecRecSys(**common_kwargs, vector_size=dense_vector_size)
        rs.train(exp.get_limited_texts(4096))
        rs.save_word2vec_format(out_fp, override=override)

    # Avg GloVe
    out_fp = models_dir / 'avg_glove.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = WeightedAvgWordVectorsRecSys(w2v_model=exp.get_w2v_model('glove'), **common_kwargs)
        rs.train(exp.texts)
        rs.save_word2vec_format(out_fp, override=override)

    # With custom GloVe embeddings
    out_fp = models_dir / 'avg_glove_custom.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = WeightedAvgWordVectorsRecSys(w2v_model=exp.get_w2v_model('glove_custom'), **common_kwargs)
        rs.train(exp.texts)
        rs.save_word2vec_format(out_fp, override=override)

    out_fp = models_dir / 'avg_fasttext.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = WeightedAvgWordVectorsRecSys(w2v_model=exp.get_w2v_model('fasttext'), **common_kwargs)
        rs.train(exp.texts)
        rs.save_word2vec_format(out_fp, override=override)

    out_fp = models_dir / 'avg_fasttext_custom.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = WeightedAvgWordVectorsRecSys(w2v_model=exp.get_w2v_model('fasttext_custom'), **common_kwargs)
        rs.train(exp.texts)
        rs.save_word2vec_format(out_fp, override=override)

    out_fp = models_dir / 'avg_fasttext_custom_512.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = WeightedAvgWordVectorsRecSys(w2v_model=exp.get_w2v_model('fasttext_custom'), **common_kwargs)
        rs.train(exp.get_limited_texts(512))
        rs.save_word2vec_format(out_fp, override=override)

    out_fp = models_dir / 'avg_fasttext_custom_4096.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = WeightedAvgWordVectorsRecSys(w2v_model=exp.get_w2v_model('fasttext_custom'), **common_kwargs)
        rs.train(exp.get_limited_texts(4096))
        rs.save_word2vec_format(out_fp, override=override)

    # Transformers
    # BERT standard pooled
    out_fp = models_dir / 'bert-base-cased.w2v.txt'
    if override or not os.path.exists(out_fp):
        rs = TransformerRecSys(model_name_or_path=env['bert_dir'] + '/bert-base-cased', **common_kwargs)
        rs.train(exp.texts)
        rs.save_word2vec_format(models_dir / 'bert-base-cased.w2v.txt', override=override)

    # All "MEAN" transformers
    for tf_name in ['bert-base-cased', 'bert-large-cased', 'roberta-base', 'roberta-large', 'legal-bert']:
        out_fp = models_dir / f'{tf_name}_mean.w2v.txt'
        if override or not os.path.exists(out_fp):
            rs = TransformerRecSys(model_name_or_path=env['bert_dir'] + '/' + tf_name,
                                   pooling_strategy='reduce_mean',
                                   **common_kwargs)
            rs.train(exp.texts)
            rs.save_word2vec_format(out_fp, override=override)

    # Long former
    if transformers.__version__ == '2.0.0':

        from longformer.longformer import Longformer
        from transformers import RobertaTokenizer

        out_fp = models_dir / 'longformer-base-4096-mean.w2v.txt'
        if override or not os.path.exists(out_fp):
            lf_lm = Longformer.from_pretrained(env['bert_dir'] + '/longformer-base-4096')
            lf_tokenizer = RobertaTokenizer.from_pretrained(env['bert_dir'] + '/roberta-base')
            lf_tokenizer.max_len = lf_lm.config.max_position_embeddings

            rs = TransformerRecSys(language_model=lf_lm, tokenizer=lf_tokenizer, max_length=4096,
                                   pooling_strategy='reduce_mean',
                                   **common_kwargs)
            rs.train(exp.texts)
            rs.save_word2vec_format(out_fp, override=override)

        out_fp = models_dir / 'longformer-large-4096-mean.w2v.txt'
        if override or not os.path.exists(out_fp):
            lf_lm = Longformer.from_pretrained(env['bert_dir'] + '/longformer-large-4096')
            lf_tokenizer = RobertaTokenizer.from_pretrained(env['bert_dir'] + '/roberta-large')
            lf_tokenizer.max_len = lf_lm.config.max_position_embeddings

            rs = TransformerRecSys(language_model=lf_lm, tokenizer=lf_tokenizer, max_length=4096,
                                   pooling_strategy='reduce_mean',
                                   **common_kwargs)
            rs.train(exp.texts)
            rs.save_word2vec_format(out_fp, override=override)
    else:
        # Wait for https://github.com/allenai/longformer/pull/14
        logger.warning('Cannot run LongFormer with transformers!=2.0.0')

    # Sentence transformer
    if LooseVersion(transformers.__version__) >= LooseVersion('2.8.0'):
        # See https://github.com/UKPLab/sentence-transformers/blob/master/requirements.txt#L1
        st_models = [
            'bert-base-nli-mean-tokens',
            'bert-large-nli-mean-tokens',
            'roberta-base-nli-mean-tokens',
            'roberta-large-nli-mean-tokens',
            'bert-base-nli-stsb-mean-tokens',
            'bert-large-nli-stsb-mean-tokens',
            'roberta-base-nli-stsb-mean-tokens',
            'roberta-large-nli-stsb-mean-tokens',
        ]
        st_dir = env['datasets_dir'] + '/sentence_transformers/'

        for st_model_name in st_models:
            out_fp = models_dir / f's{st_model_name}.w2v.txt'
            if override or not os.path.exists(out_fp):
                rs = SentenceTransformerRecSys(model_name_or_path=st_dir + st_model_name, **common_kwargs)
                rs.train(exp.texts)
                rs.save_word2vec_format(out_fp, override=override)
        #    break
    else:
        logger.warning('Cannot run sentence-transformers with transformers==%s' % transformers.__version__)

    # Citation

    # DeepWalk
    out_fp = models_dir / 'deepwalk.pickle'
    if override or not os.path.exists(out_fp):
        rs = GraphEmbeddingRecSys(
            include_seeds=exp.get_included_seeds(),
            graph_model_cls='karateclub.DeepWalk',
            graph_model_kwargs=dict(dimensions=dense_vector_size, workers=workers),
            **common_kwargs
        )
        rs.train(exp.cits)
        rs.save_to_disk(out_fp, override=override)

    # Diff2Vec
    """
    out_fp = models_dir / 'diff2vec.pickle'
    if override or not os.path.exists(out_fp):
        diff2vec = GraphEmbeddingRecSys(
            include_seeds=exp.get_included_seeds(),
            graph_model_cls='karateclub.Diff2Vec',
            graph_model_kwargs=dict(dimensions=dense_vector_size, workers=workers),
            **common_kwargs
        )
        diff2vec.train(exp.cits)
        diff2vec.save_to_disk(out_fp, override=override)
    """

    # Walklets
    out_fp = models_dir / 'walklets.pickle'
    if override or not os.path.exists(out_fp):
        walklets_window_size = 5  # or 3
        walklets_dim = int(dense_vector_size / walklets_window_size)  # must be int
        rs = GraphEmbeddingRecSys(
            include_seeds=exp.get_included_seeds(),
            graph_model_cls='karateclub.Walklets',
            graph_model_kwargs=dict(dimensions=walklets_dim, window_size=walklets_window_size, workers=workers),
            **common_kwargs
        )
        rs.train(exp.cits)
        rs.save_to_disk(out_fp, override=override)

    # Node2Vec
    out_fp = models_dir / 'node2vec.pickle'
    if override or not os.path.exists(out_fp):
        rs = GraphEmbeddingRecSys(
            include_seeds=exp.get_included_seeds(),
            graph_model_cls='node2vec.Node2Vec',
            graph_model_kwargs=dict(dimensions=dense_vector_size, workers=workers),
            **common_kwargs
        )
        rs.train(exp.cits)
        rs.save_to_disk(out_fp, override=override)

    # NodeSketch
    """
    out_fp = models_dir / 'nodesketch.pickle'
    if override or not os.path.exists(out_fp):
        nodesketch = GraphEmbeddingRecSys(
            include_seeds=exp.get_included_seeds(),
            graph_model_cls='karateclub.NodeSketch',
            graph_model_kwargs=dict(dimensions=dense_vector_size),
            **common_kwargs
        )
        nodesketch.train(exp.cits)
        nodesketch.save_to_disk(out_fp, override=override)
    """

    # BoostNE
    out_fp = models_dir / 'boostne.pickle'
    if override or not os.path.exists(out_fp):
        boostne_iters = 9  # 14
        boostne_dim = 30  # 20

        assert boostne_dim * (boostne_iters + 1) == dense_vector_size

        boostne = GraphEmbeddingRecSys(
            include_seeds=exp.get_included_seeds(),
            # vector_size=dense_vector_size,
            graph_model_cls='karateclub.BoostNE',
            graph_model_kwargs=dict(
                dimensions=boostne_dim,  # 8
                order=2,  # 2
                iterations=boostne_iters,  # 16
                alpha=0.01,
            ),
            # Take only embedding from last boosting
            # node_embedding_slice=slice(dense_vector_size * boostne_iters, dense_vector_size * (boostne_iters + 1)),
            **common_kwargs
        )
        boostne.train(exp.cits)
        boostne.save_to_disk(out_fp, override=override)

    # Poincare
    from gensim.models.poincare import PoincareModel
    out_fp = models_dir / 'poincare.w2v.txt'
    if override or not os.path.exists(out_fp):
        poincare_model = PoincareModel(
            exp.cits,
            size=300,
            alpha=0.1,
            negative=10,
            workers=1,
            epsilon=1e-05,
            regularization_coeff=1.0,
            burn_in=10, burn_in_alpha=0.01, init_range=(-0.001, 0.001),
        )
        poincare_model.train(
            epochs=50,
        )
        # init empty model
        poincare = KeyedVectors(vector_size=poincare_model.kv.vector_size)

        # ignore items not part of gold standard
        for doc_id in list(poincare_model.kv.vocab.keys()):
            if doc_id in exp.get_included_seeds():
                poincare.add(doc_id, poincare_model.kv.get_vector(doc_id))
        poincare.save_word2vec_format(out_fp)

    logger.info('Done')
