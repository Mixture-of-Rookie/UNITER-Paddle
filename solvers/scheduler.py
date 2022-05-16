#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import random
time.sleep(random.randint(0, 5))

from paddlenlp.transformers.optimization import (
    LinearDecayWithWarmup,
    ConstScheduleWithWarmup,
)


def set_scheduler(total_steps, cfg):
    lr = cfg['OPTIMIZATION']['LR']
    warmup_steps = cfg['OPTIMIZATION']['WARMUP_STEPS']
    lr_scheduler = cfg['OPTIMIZATION']['LR_SCHEDULER']

    if lr_scheduler == 'constant':
        scheduler = ConstScheduleWithWarmup(
            lr, warmup=warmup_steps)
    elif lr_scheduler == 'linear':
        scheduler = LinearDecayWithWarmup(
            lr, warmup=warmup_steps, total_steps=total_steps)
    else:
        raise ValueError('Supported schedulers are [linear, constant], \
            but the given one is {}'.format(lr_scheduler))
    return scheduler
