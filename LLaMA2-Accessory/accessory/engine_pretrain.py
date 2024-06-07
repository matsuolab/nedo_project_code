import math
import sys
from typing import Iterable
import contextlib
import os
import torch

import accessory.util.misc as misc
import accessory.util.lr_sched as lr_sched

from fairscale.nn.model_parallel import initialize as fs_init

import wandb
from accessory.util.wandb_utils import log_wandb
import logging
#import torch.distributed as torch_distributed

def train_one_epoch(model: torch.nn.Module,
                    data_loader, val_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, start_iter, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    #node_id = os.environ.get('SLURM_NODEID', 'unknown_node')
    #logging.basicConfig(filename=f'misc_node_{node_id}.log', level=logging.DEBUG)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    model.zero_grad(set_to_none=True)

    dataset_state = {}

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, item_states) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, start_iter), start=start_iter
    ):
        #torch_distributed.barrier()
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step, args)
        
        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
             c_loss, additional_loss_dict = model(examples, labels)
        loss = c_loss
        for (add_loss, weight) in additional_loss_dict.values():
            loss = loss + add_loss * weight
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        update_grad = (data_iter_step + 1) % accum_iter == 0
        #update_grad = True
        grad_norm = loss_scaler(
            loss, optimizer, model,
            parameters=model.parameters(),
            update_grad=update_grad,
            clip_grad=None if args.clip_grad <= 0 else args.clip_grad,
        )

        if update_grad:
            assert grad_norm is not None
            if torch.any(torch.isinf(grad_norm)):
                print("grad norm is inf")
            else:
                metric_logger.update(grad_norm=grad_norm)

            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # process item states for resume
        for i in range(len(item_states['worker_id'])):
            worker_id, _curr_idx, _file_idx = item_states['worker_id'][i], item_states['_curr_idx'][i], item_states['_file_idx'][i]
            worker_id, _curr_idx, _file_idx = worker_id.item(), _curr_idx.item(), _file_idx.item()
            if worker_id not in dataset_state or \
            dataset_state[worker_id]['_file_idx'] < _file_idx or \
            (dataset_state[worker_id]['_file_idx'] == _file_idx and dataset_state[worker_id]['_curr_idx'] < _curr_idx):
                dataset_state[worker_id] = {"_curr_idx": _curr_idx, "_file_idx":  _file_idx}

        # save checkpoint
        if (data_iter_step + 1) % args.save_freq == 0:
            misc.save_checkpoint(
                output_dir=args.output_dir,
                args=args, epoch=epoch, iteration=data_iter_step, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, dataset_state=dataset_state)

        # validation
        if (data_iter_step + 1) % 100000 == 0:
            val_metrics = val_one_epoch(model, val_loader, epoch, args=args)
            if log_writer is not None:
                for metric_name, metric_value in val_metrics.items():
                    log_writer.add_scalar("val"+metric_name, metric_value, data_iter_step)
            model.train(True)

        for metric_name, metric in metric_logger.meters.items():
            metric_value = metric.value
            #print("ALLreduce前",metric_name,metric_value,os.environ.get('LOCAL_RANK'))
            metric_value = misc.all_reduce_mean(metric_value, group=fs_init.get_data_parallel_group())
            #print("ALLreduce後",metric_name,metric_value,os.environ.get('LOCAL_RANK'))
            if log_writer is not None:
                log_writer.add_scalar(metric_name, metric_value, data_iter_step)
        
        #wandb
        if int(os.environ.get('LOCAL_RANK'))==0:
            print_wandb=10
            if(data_iter_step+1)%print_wandb == 0:
                wandb_lbl = 0
                for key, val in additional_loss_dict.items():
                    wandb_lbl = val[0].item()
                for k, meter in metric_logger.meters.items():
                    if k=="lr":
                        wandb_lr = meter
                log_wandb(model,c_loss_value,wandb_lbl,data_iter_step,wandb_lr.value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@ torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  data_loader: Iterable, epoch: int,
                  args=None):
    print("!!!start validation!!!")
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data_iter_step, (examples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
             c_loss, additional_loss_dict = model(examples, labels)
        c_loss_value = c_loss.item()

        metric_logger.update(closs=c_loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
