'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import logging
from metrics.metrics import compute_scores
from modules.tokenizers import Tokenizer
from modules.dataloaders import AHPDataLoader

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.ahp import ahp_decoder
import utils
from utils import cosine_lr_schedule, warmup_lr_schedule
from data.utils import save_result
import wandb

# os.environ['WANDB_MODE'] = 'offline'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, reports_ids, reports_masks = images.to(device), reports_ids.to(device), reports_masks.to(
            device)

        if epoch > 0:
            alpha = 0.4
        else:
            alpha = 0.4 * min(1, batch_idx / len(data_loader))

        loss = model(images, reports_ids, reports_masks, alpha)

        if torch.distributed.get_rank() == 0:
            wandb.log({"loss": loss})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, metric_ftns, data_loader, device, config):
    # evaluate
    logger = logging.getLogger(__name__)
    logger.info('Start to evaluate in the test set.')
    log = dict()
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    with torch.no_grad():
        test_gts, test_res = [], []
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

            images = images.to(device)

            output = model.generate(images, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                      min_length=config['min_length'])
            reports = model.tokenizer.decode_batch(output.cpu().numpy())
            # print(reports)
            ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            test_res.extend(reports)
            test_gts.extend(ground_truths)

            for report, img_id in zip(reports, images_id):
                result.append({"image_id": img_id, "report": report})


        test_met = metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                               {i: [re] for i, re in enumerate(test_res)})

        log.update(**{'test_' + k: v for k, v in test_met.items()})
        print(log)

    return log, result

def main(args, config):

    utils.init_distributed_mode(args)

    if torch.distributed.get_rank() == 0:
        if not args.wandb_id:  # 如果没有输入就重新生成
            args.wandb_id = wandb.util.generate_id()
        wandb.init(
            project="Medical Report Generation",
            config=args,
            name=config['datasets'],
            id=args.wandb_id,
            # resume = True,
            # group="Caption"
        )

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating captioning dataset")
    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = AHPDataLoader(args, tokenizer, split='train', shuffle=False, drop_last=True)
    val_dataloader = AHPDataLoader(args, tokenizer, split='val', shuffle=False, drop_last=False)
    test_dataloader = AHPDataLoader(args, tokenizer, split='test', shuffle=False, drop_last=False)

    #### Model ####
    print("Creating model")
    model = ahp_decoder(args=args, pretrained=config['pretrained'], prompt=config['prompt'], datasets=config["datasets"], tokenizer=tokenizer)
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    metrics = compute_scores

    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            # if args.distributed:
            #     train_dataloader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_dataloader, optimizer, epoch, device)

        val_log, val_result = evaluate(model_without_ddp, metrics, val_dataloader, device, config)

        test_log, test_result = evaluate(model_without_ddp, metrics, test_dataloader, device, config)

        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }

        if test_log['test_BLEU_4'] > best:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            best = test_log['test_BLEU_4']
            best_epoch = epoch
            torch.save(save_obj, os.path.join(args.output_dir,  f'checkpoint_best_{current_time}.pth'))

        if utils.is_main_process():
                wandb.log({'test/Bleu_1': test_log['test_BLEU_1'],
                           'test/Bleu_2': test_log['test_BLEU_2'],
                           'test/Bleu_3': test_log['test_BLEU_3'],
                           'test/Bleu_4': test_log['test_BLEU_4'],
                           'test/CIDEr_test': test_log['test_CIDEr'],
                           'test/ROUGE_L': test_log['test_ROUGE_L'],
                           'test/METEOR': test_log['test_METEOR'],
                           'epoch': epoch,
                           })
                wandb.log({'validation/Bleu_1': val_log['test_BLEU_1'],
                           'validation/Bleu_2': val_log['test_BLEU_2'],
                           'validation/Bleu_3': val_log['test_BLEU_3'],
                           'validation/Bleu_4': val_log['test_BLEU_4'],
                           'validation/CIDEr_test': val_log['test_CIDEr'],
                           'validation/ROUGE_L': val_log['test_ROUGE_L'],
                           'validation/METEOR': val_log['test_METEOR'],
                           'epoch': epoch,
                           })
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_log.items()},
                             **{f'test_{k}': v for k, v in test_log.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def count_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params = param.numel()
            total_params += layer_params
            print(f"{name}: {layer_params}")
    return total_params




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_mimic.yaml')
    parser.add_argument('--output_dir', default='./AHP/output/caption_iu_xray_ft')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=46, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument("--wandb_id", type=str)
    parser.add_argument('--bos_idx', type=int, default=1, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=2, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr', 'bladder'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='./data/mimic_cxr/images', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='./data/mimic_cxr/annotation.json', help='the path to the directory containing the data.')
    # parser.add_argument('--n_gpu', type=int, default=8, help='the number of gpus to be used.')


    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
