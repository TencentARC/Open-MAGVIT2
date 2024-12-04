import math
import torch
from functools import partial

# step scheduler
def fn_LinearWarmup(warmup_steps, step):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def Scheduler_LinearWarmup(warmup_steps):
    return partial(fn_LinearWarmup, warmup_steps)


def fn_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min, step):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:  # cosine learning rate schedule
        multipler = 0.5 * (math.cos((step - warmup_steps) / (max_steps - warmup_steps) * math.pi) + 1)
        return max(multipler, multipler_min)

def Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min):
    return partial(fn_LinearWarmup_CosineDecay, warmup_steps, max_steps, multipler_min)