import math
import sys
from typing import Iterable

import torch
import time

import util.misc as misc
import util.lr_sched as lr_sched


def compute_loss(latent, guide, mis_guide, labels):
    labels = torch.where(labels<100, -1.0, 1.0).to(latent.device)
    loss1 = torch.mean((latent-guide) ** 2, dim=(1,2))*(1.0+labels)
    loss2 = torch.mean((latent-mis_guide) ** 2, dim=(1,2))*(1.0-labels)
    return loss1.mean(), loss2.mean()

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    teacher_model=None,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            latent, _, _, _, _, token_drop_mask_raw, token_all_mask_raw = model(samples, train_encoder = True)
        # print(torch.max(samples), torch.min(samples), torch.mean(samples), torch.mean(samples.abs()))
        # print(model.module.mask_token_label)
        with torch.no_grad():
            teacher_model.eval()
            guide,_,_,_,_,_,_  = teacher_model(samples, train_encoder = True, token_drop_mask_raw=token_drop_mask_raw, token_all_mask_raw=token_all_mask_raw)

            if args.noise_type == 'normal':
                std = torch.std(samples)
                mis_guide_input = torch.clamp(torch.randn_like(samples)*std+torch.mean(samples), 0.0, 1.0)
            elif  args.noise_type == 'uniform':
                mis_guide_input = torch.rand_like(samples)
            elif args.noise_type == 'constant':
                mis_guide_input = torch.zeros_like(samples)+torch.mean(samples)

            
            torch.manual_seed(int(time.time()*1e10))
            mis_guide,_,_,_,_,_,_  = teacher_model(mis_guide_input, train_encoder = True, token_drop_mask_raw=token_drop_mask_raw, token_all_mask_raw=token_all_mask_raw)
            mis_guide.to(device)

        retain_l2_loss, forget_l2_loss = compute_loss(latent, guide, mis_guide, labels)
        loss = args.retain_alpha*retain_l2_loss+args.forget_alpha*forget_l2_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(retain_l2_loss=retain_l2_loss.item())
        metric_logger.update(forget_l2_loss=forget_l2_loss.item())
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        print('lr:{:.8f}\tretain_l2_loss:{:.6f}\tforget_l2_loss:{:.6f}\tloss:{:.6f}'.format(lr, retain_l2_loss,forget_l2_loss,loss))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


