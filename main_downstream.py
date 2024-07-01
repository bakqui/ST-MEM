# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.tensorboard import SummaryWriter

import models.encoder as encoder
import util.misc as misc
from engine_downstream import evaluate, train_one_epoch
from util.dataset import build_dataset, get_dataloader
from util.losses import build_loss_fn
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from util.perf_metrics import build_metric_fn, is_best_metric


def parse() -> dict:
    parser = argparse.ArgumentParser('ECG downstream training')

    parser.add_argument('--config_path',
                        required=True,
                        type=str,
                        metavar='FILE',
                        help='YAML config file path')
    parser.add_argument('--output_dir',
                        default="",
                        type=str,
                        metavar='DIR',
                        help='path where to save')
    parser.add_argument('--exp_name',
                        default="",
                        type=str,
                        help='experiment name')
    parser.add_argument('--resume',
                        default="",
                        type=str,
                        metavar='PATH',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--encoder_path',
                        default="",
                        type=str,
                        metavar='PATH',
                        help='pretrained encoder checkpoint')

    args = parser.parse_args()
    with open(os.path.realpath(args.config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config


def main(config):
    misc.init_distributed_mode(config['ddp'])

    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    device = torch.device(config['device'])

    # fix the seed for reproducibility
    seed = config['seed'] + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False

    # ECG dataset
    dataset_train = build_dataset(config['dataset'], split='train')
    dataset_valid = build_dataset(config['dataset'], split='valid')

    data_loader_train = get_dataloader(dataset_train,
                                       is_distributed=config['ddp']['distributed'],
                                       mode='train',
                                       **config['dataloader'])
    data_loader_valid = get_dataloader(dataset_valid,
                                       is_distributed=config['ddp']['distributed'],
                                       dist_eval=config['train']['dist_eval'],
                                       mode='eval',
                                       **config['dataloader'])

    if misc.is_main_process() and config['output_dir']:
        output_dir = os.path.join(config['output_dir'], config['exp_name'])
        os.makedirs(output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=output_dir)
    else:
        output_dir = None
        log_writer = None

    model_name = config['model_name']
    if model_name in encoder.__dict__:
        model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    if config['mode'] != "scratch":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['encoder_path']}")
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    if config['mode'] == "linprobe":
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    print(f"Model = {model_without_ddp}")

    eff_batch_size = config['dataloader']['batch_size'] * config['train']['accum_iter'] * misc.get_world_size()

    if config['train']['lr'] is None:
        config['train']['lr'] = config['train']['blr'] * eff_batch_size / 256

    print(f"base lr: {config['train']['lr'] * 256 / eff_batch_size}")
    print(f"actual lr: {config['train']['lr']}")
    print(f"accumulate grad iterations: {config['train']['accum_iter']}")
    print(f"effective batch size: {eff_batch_size}")

    if config['ddp']['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[config['ddp']['gpu']])
        model_without_ddp = model.module

    optimizer = get_optimizer_from_config(config['train'], model_without_ddp)
    print(optimizer)
    loss_scaler = NativeScaler()
    criterion, output_act = build_loss_fn(config['loss'])
    best_loss = float('inf')
    metric_fn, best_metrics = build_metric_fn(config['metric'])
    metric_fn.to(device)

    misc.load_model(config, model_without_ddp, optimizer, loss_scaler)

    # Start training
    print(f"Start training for {config['train']['epochs']} epochs")
    start_time = time.time()
    use_amp = config['train'].get('use_amp', True)
    for epoch in range(config['start_epoch'], config['train']['epochs']):
        if config['ddp']['distributed']:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model,
                                      criterion,
                                      data_loader_train,
                                      optimizer,
                                      device,
                                      epoch,
                                      loss_scaler,
                                      log_writer,
                                      config['train'],
                                      use_amp=use_amp,
                                      )

        valid_stats, metrics = evaluate(model,
                                        criterion,
                                        data_loader_valid,
                                        device,
                                        metric_fn,
                                        output_act,
                                        use_amp=use_amp,
                                        )
        curr_loss = valid_stats['loss']
        if output_dir and curr_loss < best_loss:
            best_loss = curr_loss
            misc.save_model(config,
                            os.path.join(output_dir, 'best-loss.pth'),
                            epoch,
                            model_without_ddp,
                            optimizer,
                            loss_scaler,
                            metrics={'loss': curr_loss,
                                     **metrics})
        for metric_name, metric_class in metric_fn.items():
            curr_metric = metrics[metric_name]
            print(f"{metric_name}: {curr_metric:.3f}")
            if output_dir and is_best_metric(metric_class, best_metrics[metric_name], curr_metric):
                best_metrics[metric_name] = curr_metric
                misc.save_model(config,
                                os.path.join(output_dir, f'best-{metric_name}.pth'),
                                epoch,
                                model_without_ddp,
                                optimizer,
                                loss_scaler,
                                metrics={'loss': valid_stats['loss'],
                                         **metrics})
            print(f"Best {metric_name}: {best_metrics[metric_name]:.3f}")

        if log_writer is not None:
            log_writer.add_scalar('perf/valid_loss', curr_loss, epoch)
            for metric_name, curr_metric in metrics.items():
                log_writer.add_scalar(f'perf/{metric_name}', curr_metric, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'valid_{k}': v for k, v in valid_stats.items()},
                     **metrics,
                     'epoch': epoch}

        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, 'log.txt'), mode='a', encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

    # Start test
    if config.get('test', False) and misc.is_main_process():
        # turn off ddp for testing
        if config['ddp']['distributed']:
            torch.distributed.destroy_process_group()

        dataset_test = build_dataset(config['dataset'], split='test')
        data_loader_test = get_dataloader(dataset_test,
                                          mode='eval',
                                          **config['dataloader'])

        model = encoder.__dict__[model_name](**config['model'])

        target_metric = config['test']['target_metric']
        checkpoint_path = os.path.join(output_dir, f'best-{target_metric}.pth')
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Load trained checkpoint from: {checkpoint_path}")
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model)

        model.to(device)

        test_stats, metrics = evaluate(model,
                                       criterion,
                                       data_loader_test,
                                       device,
                                       metric_fn,
                                       output_act,
                                       use_amp=use_amp,
                                       )
        print(f"Test loss: {test_stats['loss']:.3f}")
        for metric_name, metric in metrics.items():
            print(f"{metric_name}: {metric:.3f}")

        if output_dir:
            metrics['loss'] = test_stats['loss']
            metric_df = pd.DataFrame([metrics])
            metric_df.to_csv(os.path.join(output_dir, 'test_metrics.csv'),
                             index=False,
                             float_format='%.4f')
        print('Done!')


if __name__ == '__main__':
    config = parse()
    main(config)
