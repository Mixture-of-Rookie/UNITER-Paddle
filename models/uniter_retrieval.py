#! /usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

# paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.uniter import (
    UniterModel,
    UniterPreTrainedModel,
)

class UniterForImageTextRetrievalHardNeg(UniterPreTrainedModel):
    """Finetune UNITER for image text retrieval with harg negative mining."""
    def __init__(self, config, img_dim, margin=0.2, hard_size=16):
        super().__init__(config)
        self.margin = margin
        self.hard_size = hard_size

        self.uniter = UniterModel(config, img_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)

        self.apply(self.init_weights)
    
    def init_output(self):
        """Need to be called after from pretrained."""
        self.rank_output.weight.set_value(self.itm_output.weight[:, 1:])
        self.rank_output.bias.set_value(self.itm_output.bias[1:])

    def forward(self, batch, sample_from='t', compute_loss=True):
        # expect same input_ids for all pairs
        batch_size = batch['attn_masks'].shape[0]
        input_ids = batch['input_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        if sample_from == 't':
            if input_ids.shape[0] == 1:
                batch['input_ids'] = input_ids.expand([batch_size, -1])
        elif sample_from == 'i':
            if img_feat.shape[0] == 1:
                batch['img_feat'] = img_feat.expand([batch_size, -1, -1])
            if img_pos_feat.shape[0] == 1:
                batch['img_pos_feat'] = img_pos_feat.expand([batch_size, -1, -1])
        else:
            raise ValueError()

        if self.training and compute_loss:
            with paddle.no_grad():
                self.eval()
                scores = self.compute_score(batch, compute_loss=False)
                hard_batch = self._get_hard_batch(batch, scores, sample_from)
                self.train()
            return self.compute_score(hard_batch, compute_loss=True)
        else:
            return self.compute_score(batch, compute_loss)
    
    def compute_score(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        img_feat = batch['img_feat']
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_pos_feat = batch['img_pos_feat']
        gather_index = batch['gather_index']
        attention_mask = batch['attn_masks']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)

        if compute_loss:
            # triplet loss
            rank_scores_sigmoid = F.sigmoid(rank_scores)
            sample_size = batch['sample_size']
            scores = rank_scores_sigmoid.reshape((-1, sample_size))
            pos = scores[:, :1]
            neg = scores[:, 1:]
            rank_loss = paddle.clip(self.margin + neg - pos, 0)
            return rank_loss.mean()
        else:
            return rank_scores

    def _get_hard_batch(self, batch, scores, sample_from='t'):
        batch = defaultdict(lambda: None, batch)
        img_feat = batch['img_feat']
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        hard_batch = {'sample_size': self.hard_size + 1}

        # NOTE first example is positive
        hard_indices = scores.squeeze(-1)[1:].topk(
            self.hard_size, sorted=False)[1] + 1
        indices = paddle.concat([paddle.zeros([1, ], dtype=paddle.int64),
                                 hard_indices])

        attention_mask = attention_mask.index_select(axis=0, index=indices)
        gather_index = gather_index.index_select(axis=0, index=indices)
        if position_ids.shape[0] != 1:
            position_ids = position_ids[:self.hard_size+1]

        if sample_from == 't':
            # cut to minimum padding
            max_len = attention_mask.sum(axis=1).max().item()
            max_i = max_len - input_ids.shape[1]
            attention_mask = attention_mask[:, :max_len]
            gather_index = gather_index[:, :max_len]
            img_feat = img_feat.index_select(axis=0, index=indices)[:, :max_i, :]
            img_pos_feat = img_pos_feat.index_select(axis=0, index=indices)[:, :max_i, :]
            # expect same input_ids for all pairs
            input_ids = input_ids[:self.hard_size+1]
        elif sample_from == 'i':
            input_ids = input_ids.index_select(axis=0, index=indices)
            # expect same image features for all pairs
            img_feat = img_feat[:self.hard_size+1]
            img_pos_feat = img_pos_feat[:self.hard_size+1]
        else:
            raise ValueError()

        hard_batch['input_ids'] = input_ids
        hard_batch['position_ids'] = position_ids
        hard_batch['img_feat'] = img_feat
        hard_batch['img_pos_feat'] = img_pos_feat
        hard_batch['attn_masks'] = attention_mask
        hard_batch['gather_index'] = gather_index

        return hard_batch
