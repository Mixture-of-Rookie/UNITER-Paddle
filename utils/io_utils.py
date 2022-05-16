#! /usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import json
import lmdb
import msgpack
import numpy as np
import msgpack_numpy
msgpack_numpy.patch()

from tqdm import tqdm
from collections import defaultdict
from lz4.frame import compress, decompress

# paddle
import paddle


def _fp16_to_fp32(feat_dict):
    out = {k: arr.astype(np.float32)
           if arr.dtype == np.float16 else arr
           for k, arr in feat_dict.items()}
    return out


def compute_num_bb(confs, conf_th, min_bb, max_bb):
    num_bb = max(min_bb, (confs > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return num_bb


def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.shape[0] for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[-1]
    dtype = tensors[0].dtype
    output = paddle.zeros((bs, max_len, hid), dtype=dtype)
    if pad:
        output.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output[i, :l, ...] = t
    return output

def pad_sequence(sequences, batch_first=False, padding_value=0):
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + tuple(trailing_dims)
    else:
        out_dims = (max_len, len(sequences)) + tuple(trailing_dims)

    out_tensor = paddle.full(shape=out_dims, dtype=sequences[0].dtype, fill_value=padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

# paddle
def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:  # 最后一维
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            dim_index = paddle.expand(paddle.reshape(paddle.arange(x.shape[k], dtype=index.dtype), reshape_shape),
                                      index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0])
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


def get_ids_and_lens(db):
    lens = []
    ids = []
    for id_ in list(db.id2len.keys()):
        lens.append(db.id2len[id_])
        ids.append(id_)
    return lens, ids


def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = paddle.arange(0, out_size, dtype='int64',
                                ).unsqueeze(0).tile([batch_size, 1])

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index[i, tl:tl+nbb] = paddle.arange(max_len, max_len+nbb,
                                                   dtype='int64')
    return gather_index


class DetectFeatLmdb(object):# {{{
    def __init__(self, img_dir, conf_th=0.2, max_bb=100, min_bb=10, num_bb=36,
                 compress=False, distributed=False):
        self.img_dir = img_dir
        if conf_th == -1:
            db_name = f'feat_numbb{num_bb}'
            self.name2nbb = defaultdict(lambda: num_bb)
        else:
            db_name = f'feat_th{conf_th}_max{max_bb}_min{min_bb}'
            nbb = f'nbb_th{conf_th}_max{max_bb}_min{min_bb}.json'
            if not os.path.exists(f'{img_dir}/{nbb}'):
                # nbb is not pre-computed
                self.name2nbb = None
            else:
                self.name2nbb = json.load(open(f'{img_dir}/{nbb}'))
        self.compress = compress
        if compress:
            db_name += '_compressed'

        if self.name2nbb is None:
            if compress:
                db_name = 'all_compressed'
            else:
                db_name = 'all'
        # only read ahead on single node training
        self.env = lmdb.open(f'{img_dir}/{db_name}',
                             readonly=True, create=False,
                             readahead=not distributed)
        self.txn = self.env.begin(buffers=True)
        if self.name2nbb is None:
            self.name2nbb = self._compute_nbb()

    def _compute_nbb(self):
        name2nbb = {}
        fnames = json.loads(self.txn.get(key=b'__keys__').decode('utf-8'))
        for fname in tqdm(fnames, desc='reading images'):
            dump = self.txn.get(fname.encode('utf-8'))
            if self.compress:
                with io.BytesIO(dump) as reader:
                    img_dump = np.load(reader, allow_pickle=True)
                    confs = img_dump['conf']
            else:
                img_dump = msgpack.loads(dump, raw=False)
                confs = img_dump['conf']
            name2nbb[fname] = compute_num_bb(confs, self.conf_th,
                                             self.min_bb, self.max_bb)
        return name2nbb

    def __del__(self):
        self.env.close()

    def get_dump(self, file_name):
        # hack for MRC
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = _fp16_to_fp32(img_dump)
        else:
            img_dump = msgpack.loads(dump, raw=False)
            img_dump = _fp16_to_fp32(img_dump)
        img_dump = {k: arr[:nbb, ...] for k, arr in img_dump.items()}
        return img_dump

    def __getitem__(self, file_name):
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = {'features': img_dump['features'],
                            'norm_bb': img_dump['norm_bb']}
        else:
            img_dump = msgpack.loads(dump, raw=False)
        img_feat = paddle.to_tensor(img_dump['features'][:nbb, :], dtype='float32')
        img_bb = paddle.to_tensor(img_dump['norm_bb'][:nbb, :], dtype='float32')
        return img_feat, img_bb# }}}


class TxtLmdb(object):# {{{
    def __init__(self, db_dir, readonly=True, distributed=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False,
                                 readahead=not distributed)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret# }}}


class TxtTokLmdb(object):# {{{
    def __init__(self, db_dir, max_txt_len=60):
        if max_txt_len == -1:
            self.id2len = json.load(open(f'{db_dir}/id2len.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(open(f'{db_dir}/id2len.json')
                                           ).items()
                if len_ <= max_txt_len
            }
        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump

    def combine_inputs(self, *inputs):
        input_ids = [self.cls_]
        for ids in inputs:
            input_ids.extend(ids + [self.sep])
        return paddle.to_tensor(input_ids)

    @property
    def txt2img(self):
        txt2img = json.load(open(f'{self.db_dir}/txt2img.json'))
        return txt2img

    @property
    def img2txts(self):
        img2txts = json.load(open(f'{self.db_dir}/img2txts.json'))
        return img2txts# }}}
