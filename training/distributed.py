from contextlib import nullcontext
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def initialize(device):
    size = int(os.getenv('WORLD_SIZE', '1'))
    if device == 'cuda':
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    if size > 1:
        dist.init_process_group('nccl' if device == 'cuda' else 'gloo', timeout=timedelta(minutes=30))


def barrier():
    if dist.is_initialized():
        if dist.get_backend() == 'nccl':
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def broadcast(value):
    if dist.is_initialized():
        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]
    return value


def stop_requested(value, device):
    flag = torch.tensor(int(value), device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


class Objective(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, scales):
        from training.pretrain import training_loss
        return training_loss(self.model, x, y, scales)


def distributed_loss(model, recipe):
    objective = Objective(model)
    if recipe['compile'] and recipe['device'] == 'cuda':
        # Per-bucket graph capture can wait on NCCL work from another compiling rank.
        torch._dynamo.config.optimize_ddp = False
        objective = torch.compile(objective, fullgraph=True, mode=recipe['compile_mode'])
    return DistributedDataParallel(objective,
        device_ids=[torch.cuda.current_device()] if recipe['device'] == 'cuda' else None,
        broadcast_buffers=False)


def accumulation_context(loss_fn, final):
    return loss_fn.no_sync() if isinstance(loss_fn, DistributedDataParallel) and not final else nullcontext()
