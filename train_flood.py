"""
Unified training script for flood mapping datasets (UAVflood, SARflood, GFflood).

Usage:
    python train_flood.py -config configs/config_UAVflood.cfg
    python train_flood.py -config configs/config_SARflood.cfg
    python train_flood.py -config configs/config_GFflood.cfg
"""
import datetime
import argparse
import torch
import random
import numpy as np
from configs.config import MyConfiguration
from Trainer import Trainer
from Tester import Tester
from data.dataset_tiff import FloodDatasetTiff
from torch.utils.data import DataLoader
from models import FCN_backbone


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def for_train(model, config, args, train_data_loader, valid_data_loader,
              begin_time, resume_file):
    trainer = Trainer(
        model=model, config=config, args=args,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        begin_time=begin_time,
        resume_file=resume_file)
    trainer.train()
    print(" Training Done ! ")


def for_test(model, config, args, test_data_loader, class_name,
             begin_time, resume_file):
    tester = Tester(
        model=model, config=config, args=args,
        test_data_loader=test_data_loader,
        class_name=class_name,
        begin_time=begin_time,
        resume_file=resume_file)
    tester.eval_and_predict()
    print(" Evaluation Done ! ")


def main(config, args):
    out_channel = config.nb_classes if config.loss == 'crossentropy' else 1

    model = FCN_backbone.FCNs_VGG_ASPP_4conv1(
        config.input_channel, out_channel,
        backbone='vgg16_bn',
        pretrained=False)

    if hasattr(model, 'name'):
        config.config.set("Directory", "model_name", model.name)

    train_dataset = FloodDatasetTiff(config=config, args=args, subset='train')
    valid_dataset = FloodDatasetTiff(config=config, args=args, subset='val')
    test_dataset = FloodDatasetTiff(config=config, args=args, subset='test')

    g = torch.Generator()
    g.manual_seed(config.random_seed)

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.threads,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True)
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        pin_memory=True,
        num_workers=args.threads,
        drop_last=False)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        pin_memory=True,
        num_workers=args.threads,
        drop_last=False)

    begin_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if config.use_gpu:
        model = model.cuda(device=args.gpu)

    for_train(
        model=model, config=config, args=args,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        begin_time=begin_time,
        resume_file=args.weight)

    for_test(
        model=model, config=config, args=args,
        test_data_loader=test_data_loader,
        class_name=test_dataset.class_names,
        begin_time=begin_time,
        resume_file=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flood Mapping Training")
    parser.add_argument('-config', type=str, required=True,
                        help='path to config file (e.g. configs/config_UAVflood.cfg)')
    parser.add_argument('-input', type=str, default=None,
                        help='override root_dir from config')
    parser.add_argument('-output', type=str, default=None,
                        help='override save_dir from config')
    parser.add_argument('-weight', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('-threads', type=int, default=None,
                        help='number of DataLoader workers')
    parser.add_argument('-gpu', type=int, default=0,
                        help='gpu id (-1 for cpu)')
    parser.add_argument('-only_prediction', action='store_true', default=False)
    parser.add_argument('-is_test', action='store_true', default=False)
    pre_args, _ = parser.parse_known_args()

    config = MyConfiguration(pre_args.config)

    args = parser.parse_args()
    if args.input is None:
        args.input = config.root_dir
    if args.output is None:
        args.output = config.save_dir
    if args.threads is None:
        args.threads = config.num_workers
    if not config.use_gpu:
        args.gpu = -1

    if config.use_seed:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    else:
        torch.backends.cudnn.benchmark = True

    main(config=config, args=args)
