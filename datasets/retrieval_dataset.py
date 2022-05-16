#! /usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import random
from cytoolz import concat
from toolz.sandbox import unzip
from collections import defaultdict

# paddle
import paddle
from paddle.io import Dataset

# utils
from utils.io_utils import (
    pad_tensors,
    pad_sequence,
    get_ids_and_lens,
    get_gather_index,
)


def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """Random and retry."""
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


class DetectFeatTxtTokDataset(Dataset):#{{{
    def __init__(self, txt_db, img_db):
        self.txt_db = txt_db
        self.img_db = img_db
        txt_lens, self.ids = get_ids_and_lens(txt_db)

        txt2img = txt_db.txt2img
        self.lens = [tl + self.img_db.name2nbb[txt2img[id_]]
                     for tl, id_ in zip(txt_lens, self.ids)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        return example

    def _get_img_feat(self, fname):
        img_feat, bb = self.img_db[fname]
        img_bb = paddle.concat([bb, bb[:, 4:5]*bb[:, 5:]], axis=-1)
        num_bb = img_feat.shape[0]
        return img_feat, img_bb, num_bb#}}}


class ItmRankDataset(DetectFeatTxtTokDataset):#{{{
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, \
            "ItmRankDataset need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        # images partitioned by rank
        self.img2txts = defaultdict(list)
        for id_, img in self.txt2img.items():
            self.img2txts[img].append(id_)
        self.img_name_list = list(self.img2txts.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        id_pairs = [(gt_txt_id, gt_img_fname)]
        # sample negatives
        neg_sample_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        neg_sample_txt_ids = sample_negative(
            self.ids, self.img2txts[gt_img_fname], self.neg_sample_size)
        id_pairs.extend([(gt_txt_id, neg_img_id)
                         for neg_img_id in neg_sample_img_ids] +
                        [(neg_txt_id, gt_img_fname)
                         for neg_txt_id in neg_sample_txt_ids])
        inputs = self._collect_inputs(id_pairs)
        assert len(inputs) == (1 + 2*self.neg_sample_size)
        return inputs

    def _collect_inputs(self, id_pairs):
        # create input features
        inputs = []
        for txt_id, img_id in id_pairs:
            example = self.txt_db[txt_id]
            # text input
            input_ids = example['input_ids']
            input_ids = self.txt_db.combine_inputs(input_ids)
            # img input
            img_feat, img_pos_feat, num_bb = self._get_img_feat(img_id)
            # mask
            attn_masks = paddle.ones(len(input_ids) + num_bb, dtype='int64')
            inputs.append((input_ids, img_feat, img_pos_feat, attn_masks))
        return inputs#}}}


def itm_rank_collate(inputs):#{{{
    (input_ids, img_feats, img_pos_feats, attn_masks,) = \
        map(list, unzip(concat(i for i in inputs)))

    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = paddle.arange(0, input_ids.shape[1], dtype='int64').unsqueeze(0)

    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)

    bs, max_tl = input_ids.shape
    out_size = attn_masks.shape[0]
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    return sample_size, \
            (img_feat, input_ids, attn_masks, gather_index, position_ids, img_pos_feat)#}}}


class ItmRankDatasetHardNegFromText(DetectFeatTxtTokDataset):#{{{
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "Need at least 1 negative sample."
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.img_name_list = list(self.img2txts.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]
        
        input_ids = self.txt_db[gt_txt_id]['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0)
        position_ids = paddle.arange(0, input_ids.shape[1],
            dtype='int64').unsqueeze(0)

        neg_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        img_ids = [gt_img_fname] + neg_img_ids
        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.shape[1]
        attn_masks = paddle.zeros((len(img_ids), max(num_bbs) + tl), dtype='int64')
        for i, nbb in enumerate(num_bbs):
            attn_masks[i, :tl+nbb] = 1
        out_size = attn_masks.shape[1]
        gather_index = get_gather_index([tl]*len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index}
        return batch#}}}


class ItmRankDatasetHardNegFromImage(DetectFeatTxtTokDataset):#{{{
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db)
	
        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.txt_name_list = list(self.txt2img.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]
        gt_txt_ids = self.img2txts[gt_img_id]

        # process image features (gt always first)
        img_feat, img_pos_feat, nbb = self._get_img_feat(gt_img_id)
        img_feat = img_feat.unsqueeze(0)
        img_pos_feat = img_pos_feat.unsqueeze(0)

        # sample negative
        neg_txt_ids = sample_negative(
            self.txt_name_list, gt_txt_ids, self.neg_sample_size)
        txt_ids = [gt_txt_id] + neg_txt_ids

        # process text inputs
        all_inputs = []
        txt_lens = []
        for txt_id in txt_ids:
            input_ids = self.txt_db.combine_inputs(
                self.txt_db[txt_id]['input_ids'])
            all_inputs.append(input_ids)
            txt_lens.append(len(input_ids))
        input_ids = pad_sequence(all_inputs, batch_first=True, padding_value=0)
        position_ids = paddle.arange(0, input_ids.shape[1], dtype='int64'
                                    ).unsqueeze(0)

        attn_masks = paddle.zeros((len(txt_ids), max(txt_lens) + nbb), dtype='int64')
        for i, tl in enumerate(txt_lens):
            attn_masks[i, :tl+nbb] = 1
        out_size = attn_masks.shape[1]
        gather_index = get_gather_index(txt_lens, [nbb]*len(txt_ids),
                                        len(txt_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index}
        return batch#}}}


def itm_rank_hn_collate(inputs):# {{{
    assert len(inputs) == 1
    return inputs[0]# }}}


class ItmValDataset(DetectFeatTxtTokDataset):#{{{
    """For evaluating Image-Text-Retrieval task, i.e., dev mode."""
    def __init__(self, db_dir, img_dir, mini_batch_size=400):
        super().__init__(db_dir, img_dir)
        del self.lens
        self.txt2img = self.txt_db.txt2img
        self.img2txts = self.txt_db.img2txts
        self.all_img_ids = list(self.img2txts.keys())

        assert len(self.img2txts) >= mini_batch_size > 0
        self.bs = mini_batch_size

    def _get_batch_ids(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]

        # sample fixed negatives for each gt image
        i = self.all_img_ids.index(gt_img_id)
        neg_st = i + 1
        neg_end = neg_st + self.bs-1
        if neg_end > len(self.all_img_ids):
            # warp around
            neg_end -= len(self.all_img_ids)
            neg_img_ids = (self.all_img_ids[neg_st:]
                           + self.all_img_ids[:neg_end])
        else:
            neg_img_ids = self.all_img_ids[neg_st:neg_end]

        assert len(neg_img_ids) == (self.bs - 1),\
            "Did not sample enough neg samples"

        return gt_img_id, neg_img_ids

    def __getitem__(self, i):
        """Returns list of mini-batches."""
        gt_img_id, neg_img_ids = self._get_batch_ids(i)
        # NOTE 1st one is gt img
        batch = self.get_batch(i, [gt_img_id] + neg_img_ids)
        return batch

    def get_batch(self, i, img_ids):
        example = super().__getitem__(i)

        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0).expand((len(img_ids), -1)).clone()
        position_ids = paddle.arange(0, input_ids.shape[1], dtype='int64').unsqueeze(0)

        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.shape[1]
        attn_masks = paddle.zeros((len(img_ids), max(num_bbs) + tl), dtype='int64')
        for i, nbb in enumerate(num_bbs):
            attn_masks[i, :tl+nbb] = 1
        out_size = attn_masks.shape[1]
        gather_index = get_gather_index([tl]*len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index}
        return batch#}}}


def itm_val_collate(inputs):#{{{
    assert len(inputs) == 1, "input batch size > 1"
    return inputs[0]#}}}


class ItmEvalDataset(ItmValDataset):#{{{
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.img_db.name2nbb[i])

    def __getitem__(self, i):
        mini_batches = []
        for st in range(0, len(self.all_img_ids), self.bs):
            mini_batches.append(
                self.get_batch(i, self.all_img_ids[st:st+self.bs]))
        return mini_batches#}}}

itm_eval_collate = itm_val_collate
