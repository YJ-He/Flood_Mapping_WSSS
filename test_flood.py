"""
Unified testing script for flood mapping datasets (UAVflood, SARflood, GFflood).

Usage:
    python test_flood.py -config configs/config_UAVflood.cfg -weight path/to/checkpoint-best.pth
    python test_flood.py -config configs/config_SARflood.cfg -weight path/to/checkpoint-best.pth
    python test_flood.py -config configs/config_GFflood.cfg -weight path/to/checkpoint-best.pth
"""
import datetime
import argparse
import torch
import random
import numpy as np
from configs.config import MyConfiguration
from Tester import Tester
from data.dataset_tiff import FloodDatasetTiff
from torch.utils.data import DataLoader
from models import FCN_backbone


def main(config, args):
    out_channel = config.nb_classes if config.loss == 'crossentropy' else 1

    model = FCN_backbone.FCNs_VGG_ASPP_4conv1(
        config.input_channel, out_channel,
        backbone='vgg16_bn',
        pretrained=False)

    if hasattr(model, 'name'):
        config.config.set("Directory", "model_name", model.name)

    test_dataset = FloodDatasetTiff(config=config, args=args, subset='test')

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

    tester = Tester(
        model=model, config=config, args=args,
        test_data_loader=test_data_loader,
        class_name=test_dataset.class_names,
        begin_time=begin_time,
        resume_file=args.weight)

    if args.only_prediction:
        tester.predict()
    else:
        tester.eval_and_predict()
    print(" Testing Done ! ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flood Mapping Testing")
    parser.add_argument('-config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('-weight', type=str, required=True,
                        help='path to model checkpoint (.pth)')
    parser.add_argument('-input', type=str, default=None,
                        help='override root_dir from config')
    parser.add_argument('-output', type=str, default=None,
                        help='override save_dir from config')
    parser.add_argument('-threads', type=int, default=None,
                        help='number of DataLoader workers')
    parser.add_argument('-gpu', type=int, default=0,
                        help='gpu id (-1 for cpu)')
    parser.add_argument('-only_prediction', action='store_true', default=False)
    parser.add_argument('-is_test', action='store_true', default=True)
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
        torch.manual_seed(config.random_seed)
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    main(config=config, args=args)
