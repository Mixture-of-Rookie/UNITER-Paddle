#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
import argparse
import numpy as np

# paddle
import paddle
import paddle.distributed as dist
from paddle.io import (
    DataLoader,
    BatchSampler,
    DistributedBatchSampler,
)
# model
from models.uniter_retrieval import UniterForImageTextRetrievalHardNeg
# dataset
from datasets.retrieval_dataset import (
    itm_eval_collate,
    itm_rank_hn_collate,
    ItmEvalDataset,
    ItmRankDatasetHardNegFromText,
    ItmRankDatasetHardNegFromImage,
)
# solver
from solvers.optimizer import set_optimizer
from solvers.scheduler import set_scheduler
# config
from config.default import get_cfg_defaults
# utils
from utils.utils import (
    dump_cfg,
    compute_ranks,
    set_seed_logger,
)
from utils.io_utils import (
    TxtTokLmdb,
    DetectFeatLmdb,
)

# logging basic config
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)

def train_epoch(model, trn_dataloader_img, trn_dataloader_txt, optimizer, scheduler, epoch, logger, args, cfg):# {{{
    # Set mode for training
    model.train()
    # Set epoch for trn_sampler
    trn_dataloader_img.batch_sampler.set_epoch(epoch)
    trn_dataloader_txt.batch_sampler.set_epoch(epoch)

    logger.info('=====> Start epoch {}:'.format(epoch + 1))

    print_steps = cfg['MONITOR']['PRINT_FREQ']
    grad_accum_steps = cfg['OPTIMIZATION']['GRADIENT_ACCUMULATION_STEPS']

    train_loss = 0.
    optim_steps = 0
    train_iter_img = iter(trn_dataloader_img)
    for step, batch_txt in enumerate(trn_dataloader_txt):
        # hard text from image
        try:
            batch_img = next(train_iter_img)
        except StopIteration:
            train_iter_img = iter(trn_dataloader_img)
            batch_img = next(train_iter_img)

        # Forward img
        loss_img = model(batch_img, sample_from='i', compute_loss=True)
        if args.n_gpus > 1:
            loss_img = loss_img.mean()
        if grad_accum_steps > 1:
            loss_img = loss_img / grad_accum_steps
        # Backward img
        loss_img.backward()

        # Forward txt
        loss_txt = model(batch_txt, sample_from='t', compute_loss=True)
        if args.n_gpus > 1:
            loss_txt = loss_txt.mean()
        if grad_accum_steps > 1:
            loss_txt = loss_txt / grad_accum_steps
        # Backward txt
        loss_txt.backward()

        train_loss += (float(loss_img) + float(loss_txt))

        # Update parameters
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.clear_grad()

            # Count optimization steps
            optim_steps += 1

            # Print log
            if optim_steps % print_steps == 0:
                logger.info('Epoch [%d], step [%d], training loss: %.5f' % (
                            epoch + 1, optim_steps, (float(loss_img) + float(loss_txt))))

    train_loss = train_loss / step
    logger.info('** ** Epoch [%d] done! Training loss: %.5f ** **'
                 % (epoch + 1, train_loss))# }}}


@paddle.no_grad()
def validate(model, dev_dataloader, epoch, logger):# {{{
    # Set mode for evaluation
    model.eval()

    score_matrix = paddle.zeros((len(dev_dataloader.dataset),
                                len(dev_dataloader.dataset.all_img_ids)))

    for i, mini_batches in enumerate(dev_dataloader):
        j = 0
        for batch in mini_batches:
            scores = model(batch, compute_loss=False)
            bs = scores.shape[0]
            score_matrix[i, j:j+bs] = scores.squeeze(1)
            j += bs
        assert j == score_matrix.shape[1]

    dev_dataset = dev_dataloader.dataset
    all_txt_ids = [ids for ids in dev_dataset.ids]
    all_img_ids = dev_dataset.all_img_ids
    assert score_matrix.shape == [len(all_txt_ids), len(all_img_ids)]

    dev_results = compute_ranks(score_matrix, all_txt_ids, all_img_ids,
                                dev_dataset.txt2img, dev_dataset.img2txts)

    logger.info('** * ** Eval at Epoch [%d]! Eval Reults:  ** * **' % (epoch + 1))
    logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
        dev_results['txt_r1'], dev_results['txt_r5'], dev_results['txt_r10']))
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
        dev_results['img_r1'], dev_results['img_r5'], dev_results['img_r10']))# }}}


def main(args, cfg):
    # 1. Preparation
    logger, save_dir = set_seed_logger(args, cfg)
    # backup config
    cfg_file = os.path.join(save_dir, 'config.yaml')
    dump_cfg(cfg, cfg_file)

    # 2. Create train/dev dataloader
    trn_img_db = DetectFeatLmdb(img_dir=cfg['DATASET']['IMG_DIR'])
    trn_txt_dr = os.path.join(cfg['DATASET']['TXT_DIR'], 'itm_flickr30k_{}.db'.format(cfg['DATASET']['TRAIN']))
    trn_txt_db = TxtTokLmdb(db_dir=trn_txt_dr, max_txt_len=cfg['INPUT']['MAX_SEQ_LEN'])
    # dataset
    trn_dataset_txt = ItmRankDatasetHardNegFromText(trn_txt_db, trn_img_db, neg_sample_size=cfg['INPUT']['NEG_SIZE'])
    trn_dataset_img = ItmRankDatasetHardNegFromImage(trn_txt_db, trn_img_db, neg_sample_size=cfg['INPUT']['NEG_SIZE'])
    # datasampler
    trn_sampler_txt = DistributedBatchSampler(dataset=trn_dataset_txt,
                                              batch_size=1,
                                              shuffle=True,
                                              drop_last=True)
    trn_sampler_img = DistributedBatchSampler(dataset=trn_dataset_img,
                                              batch_size=1,
                                              shuffle=True,
                                              drop_last=True)
    # dataloader
    trn_dataloader_txt = DataLoader(trn_dataset_txt,
                                    batch_sampler=trn_sampler_txt,
                                    collate_fn=itm_rank_hn_collate,
                                    num_workers=cfg['MISC']['NUM_WORKERS'])
    trn_dataloader_img = DataLoader(trn_dataset_img,
                                    batch_sampler=trn_sampler_img,
                                    collate_fn=itm_rank_hn_collate,
                                    num_workers=cfg['MISC']['NUM_WORKERS'])

    dev_img_db = DetectFeatLmdb(img_dir=cfg['DATASET']['IMG_DIR'])
    dev_txt_dr = os.path.join(cfg['DATASET']['TXT_DIR'], 'itm_flickr30k_{}.db'.format(cfg['DATASET']['DEV']))
    dev_txt_db = TxtTokLmdb(db_dir=dev_txt_dr, max_txt_len=-1)
    dev_dataset = ItmEvalDataset(dev_txt_db, dev_img_db, mini_batch_size=cfg['OPTIMIZATION']['DEV_BATCH_SIZE'])
    dev_sampler = BatchSampler(dataset=dev_dataset,
                               batch_size=1,
                               shuffle=False,
                               drop_last=False)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_sampler=dev_sampler,
                                collate_fn=itm_eval_collate,
                                num_workers=cfg['MISC']['NUM_WORKERS'])

    # 3. Build model
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

    num_params = sum([np.prod(param.shape) for param in model.parameters()])
    logger.info('Total parameters: %.2f M.' % (num_params / 1e6))

    if args.n_gpus > 1:
        model = paddle.DataParallel(model)

    # 4. Set up optimizer & lr scheduler
    num_iters = len(trn_dataloader_img)
    epochs = cfg['OPTIMIZATION']['EPOCHS']
    batch_size = cfg['OPTIMIZATION']['TRN_BATCH_SIZE'] * args.n_gpus
    grad_accum_steps = cfg['OPTIMIZATION']['GRADIENT_ACCUMULATION_STEPS']
    train_optimization_steps = num_iters * epochs // grad_accum_steps

    scheduler = set_scheduler(train_optimization_steps, cfg)
    optimizer = set_optimizer(model, scheduler, cfg)

    # 6. Training
    logger.info("** ** ** Running training ** ** **")
    logger.info('Num GPUs: %d' % args.n_gpus)
    logger.info("Num Iters: %d" % num_iters)
    logger.info("Batch Size: %d" % batch_size)
    logger.info('Accum Steps: %d' % grad_accum_steps)
    logger.info("Optim Steps: %d" % train_optimization_steps)

    for epoch in range(epochs):
        # Train one epoch
        train_epoch(model, trn_dataloader_img, trn_dataloader_txt, optimizer,
                    scheduler, epoch, logger, args, cfg)

        # Perform evaluation
        #if (epoch + 1) % cfg['MONITOR']['EVAL_FREQ'] == 0:
        #    validate(model, dev_dataloader, epoch, logger)

        if (epoch + 1) % cfg['OUTPUT']['SAVE_FREQ'] == 0:
            if dist.get_rank() == 0:
                checkpoint_dir = os.path.join(save_dir, 'checkpoint-{}'.format(epoch + 1))
                if not os.path.isdir(checkpoint_dir):
                    os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, 'paddle_model.bin'.format(epoch + 1))
                paddle.save({'epoch': epoch,
                             'model': model.state_dict()}, checkpoint_path)
                logger.info('** * ** Saving trained model to {}. ** * **'.format(checkpoint_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True,
        help='Path to the config file for a specific experiment.')
    args = parser.parse_args()

    # get default config & merge from cfg_file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # Set cuda device
    paddle.set_device('gpu')

    args.n_gpus = paddle.distributed.get_world_size()
    # Distributed training
    if args.n_gpus > 1:
        paddle.distributed.init_parallel_env()

    main(args, cfg)
