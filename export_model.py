#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import sys
import numpy as np

# model
from models.uniter_retrieval import UniterForImageTextRetrievalHardNeg
# config
from config.default import get_cfg_defaults


# parse args
def get_args(add_help=True):
    """get_args
    Parse all args using argparse lib
    Args:
        add_help: Whether to add -h option on args
    Returns:
        An object which contains many parameters used for inference.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Interference Args', add_help=add_help)
    parser.add_argument('--cfg_file', type=str,
        help='Path to the config file for a specific experiment.')
    parser.add_argument('--out_dir', type=str, default='static',
        help='Path to save the static model.')
    args = parser.parse_args()

    # Get the default config & merge from cfg_file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    return args, cfg


def build_model(args, cfg):
    """build_model
    Build your own model.
    Args:
        args: Parameters generated using argparser.
    Returns:
        A model whose type is nn.Layer
    """
    config = os.path.join(cfg['PRETRAINED']['DIR'], cfg['PRETRAINED']['CONFIG'])
    checkpoint = paddle.load(os.path.join(cfg['PRETRAINED']['DIR'], cfg['PRETRAINED']['WEIGHTS']))

    # List of parameters to be transposed
    transpose_list = ['uniter.embeddings.token_type_embeddings.weight',
                      'itm_output.weight']

    for k, v in checkpoint.items():
        if k in transpose_list:
            checkpoint[k] = v.transpose((1, 0))

    model = UniterForImageTextRetrievalHardNeg.from_pretrained(
        config, checkpoint, img_dim=cfg['INPUT']['IMG_DIM'],
        margin=cfg['INPUT']['MARGIN'], hard_size=cfg['INPUT']['HARD_NEG_SIZE'])
    model.init_output()
    model.eval()
    return model


def export(args, cfg):
    """export
    export inference model using jit.save
    Args:
        args: Parameters generated using argparser.
    Returns: None
    """
    model = build_model(args, cfg)

    # decorate model with jit.save
    model = paddle.jit.to_static(
        model,
        input_spec=[
            {
            'input_ids': InputSpec(shape=[1, 40], dtype='int64'),
            'position_ids': InputSpec(shape=[1, 40], dtype='int64'),
            'img_feat': InputSpec(shape=[1, 60, 2048], dtype='float32'),
            'img_pos_feat': InputSpec(shape=[1, 60, 7], dtype='float32'),
            'attn_masks': InputSpec(shape=[1, 100], dtype='int64'),
            'gather_index': InputSpec(shape=[1, 100], dtype='int64'),
            },
            't',
            True
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(args.out_dir, "inference"))
    print(f"inference model is saved in {args.out_dir}")


if __name__ == "__main__":
    args, cfg = get_args()
    export(args, cfg)
