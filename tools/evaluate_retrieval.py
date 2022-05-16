#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
sys.path.insert(0, '.')
import argparse
from tqdm import tqdm

# paddle
import paddle
from paddle.io import DataLoader, BatchSampler

# model
from models.uniter_retrieval import UniterForImageTextRetrievalHardNeg
# dataset
from datasets.retrieval_dataset import itm_eval_collate, ItmEvalDataset
# config
from config.default import get_cfg_defaults
# utils
from utils.utils import compute_ranks
from utils.io_utils import TxtTokLmdb, DetectFeatLmdb


def main(args, cfg):
    # 1. Create test dataloader
    test_img_db = DetectFeatLmdb(img_dir=cfg['DATASET']['IMG_DIR'])
    test_txt_dr = os.path.join(cfg['DATASET']['TXT_DIR'], 'itm_flickr30k_{}.db'.format(cfg['DATASET']['TEST']))
    test_txt_db = TxtTokLmdb(db_dir=test_txt_dr, max_txt_len=-1)
    test_dataset = ItmEvalDataset(test_txt_db, test_img_db, mini_batch_size=cfg['OPTIMIZATION']['DEV_BATCH_SIZE'])
    test_sampler = BatchSampler(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_sampler=test_sampler,
                                 collate_fn=itm_eval_collate)

    # 2. Build model
    config = os.path.join(cfg['PRETRAINED']['DIR'], cfg['PRETRAINED']['CONFIG'])
    checkpoint = paddle.load(os.path.join(args.checkpoint_dir, 'paddle_model.bin'))['model']
    model = UniterForImageTextRetrievalHardNeg.from_pretrained(
        config, checkpoint, img_dim=cfg['INPUT']['IMG_DIM'],
        margin=cfg['INPUT']['MARGIN'], hard_size=cfg['INPUT']['HARD_NEG_SIZE'])
    print('Load state dict from %s.' % args.checkpoint_dir)
    model.eval()

    # 3. Start to evaluate
    score_matrix = paddle.zeros((len(test_dataloader.dataset),
                                len(test_dataloader.dataset.all_img_ids)))

    for i, mini_batches in enumerate(tqdm(test_dataloader)):
        j = 0
        for batch in mini_batches:
            with paddle.no_grad():
                scores = model(batch, compute_loss=False)
            bs = scores.shape[0]
            score_matrix[i, j:j+bs] = scores.squeeze(1)
            j += bs
        assert j == score_matrix.shape[1]

    test_dataset = test_dataloader.dataset
    all_txt_ids = [ids for ids in test_dataset.ids]
    all_img_ids = test_dataset.all_img_ids
    assert score_matrix.shape == [len(all_txt_ids), len(all_img_ids)]

    test_results = compute_ranks(score_matrix, all_txt_ids, all_img_ids,
                                 test_dataset.txt2img, test_dataset.img2txts)

    print("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
        test_results['txt_r1'], test_results['txt_r5'], test_results['txt_r10']))
    print("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
        test_results['img_r1'], test_results['img_r5'], test_results['img_r10']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True,
        help='Path to the config file for a specific experiment.')
    args = parser.parse_args()

    # Get the default config & merge from cfg_file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # Make sure checkpoint dir exists
    args.checkpoint_dir = cfg['EVAL']['CHECKPOINT_DIR']
    assert os.path.isdir(args.checkpoint_dir), \
        "Please make sure the specified checkpoint dir and eval epoch exist."

    # Call main
    main(args, cfg)
