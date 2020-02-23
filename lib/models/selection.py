import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import json
import os
import copy
from lib.utils.pretrain import PretranEmbedding, load_pretrained_embedding
import random
import math 
from typing import Dict, List, Tuple, Set, Optional
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from lib.models.CRF import CRF
from pytorch_transformers import *
import numpy as np 

class MultiHeadSelection(nn.Module):
    def __init__(self, hyper) -> None:
        super(MultiHeadSelection, self).__init__()

        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))
        use_pretrain_embedding = True
        self.word_embeddings = nn.Embedding(num_embeddings=len(
            self.word_vocab),
            embedding_dim=hyper.emb_size)
        if use_pretrain_embedding:
            self.pe = PretranEmbedding(self.hyper)
            self.word_embeddings.weight.data.copy_(
            load_pretrained_embedding(self.word_vocab, self.pe))
    
        self.input_dropout = nn.Dropout(p=0.5)
        self.relation_emb = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
            embedding_dim=hyper.rel_emb_size)
        if hyper.cell_name == 'gru':
            self.encoder = nn.GRU(hyper.emb_size,
                                  hyper.hidden_size,
                                  bidirectional=True,
                                  batch_first=True)
        elif hyper.cell_name == 'lstm':
            self.encoder = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.classifier = nn.Linear(100, len(self.relation_vocab))
        self.dropout_c = nn.Dropout(0.5)



    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        # tokens_len = torch.tensor(list(sample.length))
        # tokens = pad_sequence(sample.tokens_id, batch_first=True, padding_value=self.word_vocab['<pad>']).to(self.gpu)
        tokens = sample.tokens_id.cuda(self.gpu)


        # if self.hyper.cell_name in ('gru', 'lstm'):
        #     mask = tokens != self.word_vocab['<pad>']  # batch x seq

        if self.hyper.cell_name in ('lstm', 'gru'):
            embedded = self.word_embeddings(tokens)
            embedded = self.input_dropout(embedded)
            # embedded = nn.utils.rnn.pack_padded_sequence(embedded, tokens_len, batch_first=True)
            o, h = self.encoder(embedded)
            # o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
            o = (lambda a: sum(a) / 2)(torch.split(o,
                                                   self.hyper.hidden_size,
                                                   dim=2))
            feature = torch.max(o, 2)[0]
            output = self.classifier(feature)
            output = self.dropout_c(output)

        return output


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass

