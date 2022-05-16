#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import logging
import datetime
import numpy as np

# paddle
import paddle

logger = logging.getLogger(__name__)


def get_logger(log_file=None):# {{{
    """Set logger and return it.
    If the log_file is not None, log will be written into log_file.
    Else, log will be shown in the screen.
    Args:
        log_file (str): If log_file is not None, log will be written
            into the log_file.
    Return:
        ~Logger
        * **logger**: An Logger object with customed config.
    """
    # Basic config
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Add filehandler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

    return logger# }}}

def set_seed_logger(args, cfg):# {{{
    """Experiments preparation, e.g., fix random seed, prepare checkpoint dir
    and set logger.
    Args:
        args (parser.Argument): An parser.Argument object.
        cfg (yacs.config): An yacs.config.CfgNode object.
    Return:
        ~(Logger, str):
        * **logger**: An Logger object with customed config.
        * **save_dir**: Checkpoint dir to save models.
    """
    seed = cfg['MISC']['SEED']
    # Set random seed
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Prepare save dir
    if cfg['OUTPUT']['SAVE_NAME']:
        prefix = cfg['OUTPUT']['SAVE_NAME'] + '_'
    else:
        prefix = ''
    exp_name = prefix + datetime.datetime.now().strftime('%yY_%mM_%dD_%HH')
    save_dir = os.path.join(cfg['OUTPUT']['CHECKPOINT_DIR'], exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Build logger
    log_file = os.path.join(save_dir, 'log.txt')
    logger = get_logger(log_file)

    return logger, save_dir# }}}

def dump_cfg(cfg, cfg_file):# {{{
    """Dump config of each experiment into file for backup.
    Args:
        cfg (yacs.config): An yacs.config.CfgNode object.
        cfg_file (str): Dump config to this file.
    """
    logger.info('Dump configs into {}'.format(cfg_file))
    logger.info('Using configs: ')
    logger.info(cfg)
    with open(cfg_file, 'w') as f:
        f.write(cfg.dump())# }}}

def compute_ranks(score_matrix, txt_ids, img_ids, txt2img, img2txt):
    # Image retrieval
    img2j = {i: j for j, i in enumerate(img_ids)}
    _, rank_txt = score_matrix.topk(10, axis=1)
    gt_img_j = paddle.to_tensor([img2j[txt2img[txt_id]]
                                 for txt_id in txt_ids],
                                dtype=paddle.int64
                               ).unsqueeze(1).expand_as(rank_txt)
    rank = (rank_txt == gt_img_j).nonzero()
    if rank.numel():
        ir_r1 = (rank < 1).sum().item() / len(txt_ids)
        ir_r5 = (rank < 5).sum().item() / len(txt_ids)
        ir_r10 = (rank < 10).sum().item() / len(txt_ids)
    else:
        ir_r1, ir_r5, ir_r10 = 0, 0, 0

    # Text retrieval
    txt2i = {t: i for i, t in enumerate(txt_ids)}
    _, rank_img = score_matrix.topk(10, axis=0)
    tr_r1, tr_r5, tr_r10 = 0, 0, 0
    for j, img_id in enumerate(img_ids):
        gt_is = [txt2i[t] for t in img2txt[img_id]]
        ranks = [(rank_img[:, j] == i).nonzero() for i in gt_is]
        rank = min([10] + [r.item() for r in ranks if r.numel()])
        if rank < 1:
            tr_r1 += 1
        if rank < 5:
            tr_r5 += 1
        if rank < 10:
            tr_r10 += 1
    tr_r1 /= len(img_ids)
    tr_r5 /= len(img_ids)
    tr_r10 /= len(img_ids)

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_res = {'txt_r1': tr_r1,
                'txt_r5': tr_r5,
                'txt_r10': tr_r10,
                'txt_r_mean': tr_mean,
                'img_r1': ir_r1,
                'img_r5': ir_r5,
                'img_r10': ir_r10,
                'img_r_mean': ir_mean,
                'r_mean': r_mean}
    return eval_res
