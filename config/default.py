#! /usr/bin/env python
# -*- coding: utf-8 -*-

from yacs.config import CfgNode as CN

# Create a Node
__C = CN()

# ========================== INPUT =========================
__C.INPUT = CN()
# Number of bounding boxes
__C.INPUT.NUM_BB = 36
# Minimum number of bounding boxes
__C.INPUT.MIN_BB = 10
# Maximum number of bounding boxes
__C.INPUT.MAX_BB = 100
# Threshold for dynamic bounding boxes
# -1 means fix, i.e., use NUM_BB.
__C.INPUT.CONF_TH = 0.2
# Negative size
__C.INPUT.NEG_SIZE = 1
__C.INPUT.HARD_NEG_SIZE = 1
__C.INPUT.MARGIN = 0.2
__C.INPUT.MAX_REGION = 50
__C.INPUT.MAX_SEQ_LEN = 60
__C.INPUT.IMG_DIM = 2048

# ========================== DATASET =========================
__C.DATASET = CN()
__C.DATASET.NAME = 'flickr30k'
__C.DATASET.IMG_DIR = ''
__C.DATASET.TXT_DIR = ''
__C.DATASET.TRAIN = 'train'
__C.DATASET.DEV = 'val'
__C.DATASET.TEST = 'test'

# ========================== OUPUT =========================
__C.OUTPUT = CN()
__C.OUTPUT.SAVE_NAME = ''
# Save checkpoint frequency (epochs)
__C.OUTPUT.SAVE_FREQ = 1
__C.OUTPUT.CHECKPOINT_DIR = './exp'

# ========================== OPTIMIZATION =========================
__C.OPTIMIZATION = CN()
__C.OPTIMIZATION.LR = 5e-5
__C.OPTIMIZATION.BETAS = [0.9, 0.98]
__C.OPTIMIZATION.TRN_BATCH_SIZE = 40
__C.OPTIMIZATION.DEV_BATCH_SIZE = 40
__C.OPTIMIZATION.WARMUP_STEPS = 500
__C.OPTIMIZATION.LR_SCHEDULER = 'linear'
__C.OPTIMIZATION.WEIGHT_DECAY = 0.01
__C.OPTIMIZATION.EPOCHS = 1
# Clip gradients at this value
__C.OPTIMIZATION.CLIP_MAX_NORM = 2.0
__C.OPTIMIZATION.OPTIMIZER = 'adamw'
# Gradient accumulation steps
__C.OPTIMIZATION.GRADIENT_ACCUMULATION_STEPS = 32

# ========================== MONITOR =========================
__C.MONITOR = CN()
# Evaluation frequency (epochs)
__C.MONITOR.EVAL_FREQ = 1
__C.MONITOR.PRINT_FREQ = 10

# ========================== PRETRAINED =========================
__C.PRETRAINED = CN()
__C.PRETRAINED.DIR = ''
__C.PRETRAINED.CONFIG = ''
__C.PRETRAINED.WEIGHTS = ''
__C.PRETRAINED.RESUME = ''

# ========================== EVAL =========================
__C.EVAL = CN()
__C.EVAL.CHECKPOINT_DIR = ''

# ========================== MISC =========================
__C.MISC = CN()
__C.MISC.SEED = 88
__C.MISC.NUM_WORKERS = 4


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return __C.clone()
