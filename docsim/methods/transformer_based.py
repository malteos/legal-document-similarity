import logging
from typing import List
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from docsim.methods.keyed_vector_based import KeyedVectorRecSys

stop_words = set(stopwords.words('english'))


logger = logging.getLogger(__name__)


class TransformerRecSys(KeyedVectorRecSys):
    model_name_or_path = None
    batch_size = 12
    max_length = 512
    language_model = None
    tokenizer = None
    cut_off_long_text = True
    pooling_strategy = 'pooled'  # reduce_mean, reduce_max

    def train(self, texts: List):
        # Import everything locally
        import torch
        from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
        from torch.nn.utils.rnn import pad_sequence

        from transformers import AutoTokenizer, AutoModel

        # load transformer language model
        if not self.language_model:
            self.language_model = AutoModel.from_pretrained(self.model_name_or_path)

        # reset
        self.model = KeyedVectors(vector_size=self.language_model.config.hidden_size)

        # auto determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.language_model = self.language_model.cuda()
        else:
            device = torch.device("cpu")
            logger.error('CUDA GPU is not available')

        # Load tokenizer if not set
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        # Cut off too long texts
        if self.cut_off_long_text:
            logger.info(f'Cut off too long text at {2*self.max_length}')
            texts = [' '.join(t.split()[:2*self.max_length]) for t in texts]

        # tokenize all texts
        logger.info('Tokenizer started...')
        if hasattr(self.tokenizer, 'batch_encode_plus'):
            tokenized = self.tokenizer.batch_encode_plus(texts, max_length=self.max_length, pad_to_max_length=True, return_tensors='pt')
        else:
            logger.info('Classic tokenize..')
            # classic tokenize
            input_ids = [torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)[:self.max_length]) for text in texts]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence([torch.tensor([1] * len(i)) for i in input_ids], batch_first=True, padding_value=0)
            tokenized = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

        # sequential data loader
        ds = TensorDataset(tokenized['input_ids'], tokenized['attention_mask'])
        dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=self.batch_size)

        # feed forward in batches
        doc_vecs = []
        logger.info('Feeding texts into Transformer...')

        with torch.no_grad():  # Disable gradient for evaluation
            for batch_data in self.get_progress_iterator(dl):
                input_ids, masks = tuple(t.to(device) for t in batch_data)

                # last_hidden_state, pooler_output = self.language_model(input_ids, masks)
                sequence_output, pooler_output = self.language_model(input_ids, masks)

                # get vectors depending on pooling strategy
                doc_vecs += list(self._pool_tokens(sequence_output, pooler_output, masks, self.pooling_strategy, ignore_first_token=False))

        # save into keyed vector
        for idx, vec in enumerate(doc_vecs):
            self.model.add([str(self.idx2doc_id[idx])], [vec])

        return self.model

    def _pool_tokens(self, sequence_output, pooler_output, padding_mask, strategy, ignore_first_token):
        """

        See https://github.com/deepset-ai/FARM/blob/5398d6f9e8a1ec8aa78b01ed6361ba6bb656a1bf/farm/modeling/language_model.py#L304

        :param sequence_output:
        :param padding_mask:
        :param strategy:
        :param ignore_first_token:
        :return:
        """

        if strategy == "pooled":
            pooled_vecs = pooler_output.cpu().numpy()
        else:
            token_vecs = sequence_output.cpu().numpy()
            # we only take the aggregated value of non-padding tokens
            padding_mask = padding_mask.cpu().numpy()
            ignore_mask_2d = padding_mask == 0

            # sometimes we want to exclude the CLS token as well from our aggregation operation
            if ignore_first_token:
                ignore_mask_2d[:, 0] = True

            ignore_mask_3d = np.zeros(token_vecs.shape, dtype=bool)
            ignore_mask_3d[:, :, :] = ignore_mask_2d[:, :, np.newaxis]

            if strategy == "reduce_max":
                pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).max(axis=1).data
            elif strategy == "reduce_mean":
                pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).mean(axis=1).data
            else:
                raise ValueError(f'Invalid pooling strategy: {strategy}')

        return pooled_vecs


class SentenceTransformerRecSys(KeyedVectorRecSys):
    model_name_or_path = None
    batch_size = 12
    language_model = None

    def train(self, texts: List):
        from sentence_transformers import SentenceTransformer

        # load sentence transformer model
        if not self.language_model:
            logger.info(f'Loading Sentence Transformer: {self.model_name_or_path}')
            self.language_model = SentenceTransformer(self.model_name_or_path)

        # reset doc vector model
        self.model = KeyedVectors(vector_size=self.language_model.get_sentence_embedding_dimension())

        # encode
        sentence_embeddings = self.language_model.encode(texts, batch_size=self.batch_size,
                                                         show_progress_bar=self.print_progress)

        # save into keyed vector
        for idx, vec in enumerate(sentence_embeddings):
            self.model.add([str(self.idx2doc_id[idx])], [vec])

        return self.model
