import os
import math
import torch
import shutil
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import utils.util
from utils.util import AverageMeter, ensure_dir
from utils import TreeEnergyLoss
from tqdm import tqdm
from utils.metrics import Evaluator_tensor
from torch.cuda.amp import autocast
from torch.cuda.amp import grad_scaler

class Trainer(object):

    def __init__(self,
                 model,
                 config,
                 args,
                 train_data_loader,
                 valid_data_loader,
                 begin_time,
                 resume_file=None):

        print("     + Training Start ... ...")
        # for general
        self.config = config
        self.args = args
        self.device = (self._device(self.args.gpu))
        self.model = model.to(self.device)

        self.train_data_loader = train_data_loader
        self.valid_data_loder = valid_data_loader

        # for time
        self.begin_time = begin_time  # part of ckpt name
        self.save_period = self.config.save_period  # for save ckpt
        self.dis_period = self.config.dis_period  # for display

        self.model_name = self.config.model_name

        self.checkpoint_dir = os.path.join(self.args.output, self.model_name,
                                           self.begin_time)
        self.log_dir = os.path.join(self.args.output, self.model_name,
                                    self.begin_time, 'log')

        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)

        # log file
        log_file_path = os.path.join(self.log_dir, self.model_name + '.txt')
        self.config.write_to_file(log_file_path)

        self.history = {
            'train': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'iou': [],
                'prec': [],
                'recall': [],
                'f_score': [],
            },
            'valid': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'iou': [],
                'prec': [],
                'recall': [],
                'f_score': [],
            }
        }
        # for optimize
        self.current_lr = self.config.init_lr

        # for train
        self.start_epoch = 0
        self.early_stop = self.config.early_stop  # early stop steps
        self.monitor_mode = self.config.monitor.split('/')[0]
        self.monitor_metric = self.config.monitor.split('/')[1]
        self.monitor_best = 0
        self.best_epoch = -1
        self.not_improved_count = 0
        self.monitor_iou = 0
        self.last_monitor_iou = 0
        self.monitor_epoch = 0
        self.last_monitor_epoch = 0

        # resume file: the confirmed ckpt file.
        self.resume_file = resume_file
        self.resume_ = True if resume_file else False
        if self.resume_file is not None:
            with open(log_file_path, 'a') as f:
                f.write('\n')
                f.write('resume_file:' + resume_file + '\n')
            self._resume_ckpt(resume_file=resume_file)

        self.optimizer = self._optimizer(lr_algorithm=self.config.lr_algorithm)

        # monitor init
        if self.monitor_mode != 'off':
            assert self.monitor_mode in ['min', 'max']
            self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        if self.config.use_one_cycle_lr:
            self.lr_scheduler = self._lr_scheduler_onecycle(self.optimizer)
        else:
            self.lr_scheduler = self._lr_scheduler_lambda(self.optimizer, last_epoch=self.start_epoch - 1)

        self.evaluator = Evaluator_tensor(self.config.nb_classes, self.device)

    def _device(self, gpu):

        if gpu == -1:
            device = torch.device('cpu')
            return device
        else:
            device = torch.device('cuda:{}'.format(gpu))
            return device

    def _optimizer(self, lr_algorithm):
        assert lr_algorithm in ['adam', 'adamw', 'sgd']
        if lr_algorithm == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                   lr=self.current_lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=self.config.weight_decay,
                                   amsgrad=False
                                   )
            return optimizer
        if lr_algorithm == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                  lr=self.current_lr,
                                  momentum=self.config.momentum,
                                  dampening=0,
                                  weight_decay=self.config.weight_decay,
                                  nesterov=True)
            return optimizer
        if lr_algorithm == 'adamw':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.current_lr,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=self.config.weight_decay,
                                    amsgrad=False
                                    )
            return optimizer

    def _lr_scheduler_onecycle(self, optimizer):
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.config.init_lr * 6,
                                                     steps_per_epoch=len(self.train_data_loader),
                                                     epochs=self.config.epochs + 1,
                                                     div_factor=6)
        return lr_scheduler

    def _lr_scheduler_lambda(self, optimizer, last_epoch):
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.config.epochs)), 0.9)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=last_epoch)

        return lr_scheduler

    def _loss_js_div(self, p_output, q_output, get_softmax=True):
        """
        Calculate JS divergence loss
        """
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = F.softmax(p_output, dim=1)
            q_output = F.softmax(q_output, dim=1)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

    def train(self):
        epochs = self.config.epochs
        assert self.start_epoch < epochs

        for epoch in range(self.start_epoch, epochs + 1):
            train_log = self._train_epoch(epoch)
            eval_log = self._eval_epoch(epoch)

            if not self.config.use_one_cycle_lr:
                # lr update
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch)
                    for param_group in self.optimizer.param_groups:
                        self.current_lr = param_group['lr']

            best = False
            diff = 0
            if self.monitor_mode != 'off':
                improved = (self.monitor_mode == 'min' and eval_log[
                    'val_' + self.monitor_metric] < self.monitor_best) or \
                           (self.monitor_mode == 'max' and eval_log['val_' + self.monitor_metric] > self.monitor_best)
                if improved:
                    self.monitor_best = eval_log['val_' + self.monitor_metric]
                    self.last_monitor_iou = self.monitor_iou
                    self.last_monitor_epoch = self.monitor_epoch
                    self.monitor_epoch = eval_log['epoch']
                    self.monitor_iou = eval_log['val_IoU']

                    best = True
                    self.best_epoch = eval_log['epoch']
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stop:
                    print("     + Validation Performance didn\'t improve for {} epochs."
                          "     + Training stop :/"
                          .format(self.not_improved_count))
                    break
            if epoch % self.save_period == 0 or best == True:
                self._save_ckpt(epoch, best=best)

        # save history file
        print("     + Saving History ... ... ")
        hist_path = os.path.join(self.log_dir, 'history1.txt')
        with open(hist_path, 'w') as f:
            f.write(str(self.history))

    def train_TFCSD(self):
        epochs = self.config.epochs
        assert self.start_epoch < epochs

        for epoch in range(self.start_epoch, epochs + 1):
            # get log information of train and evaluation phase
            train_log = self._train_epoch_TFCSD(epoch)
            eval_log = self._eval_epoch(epoch)

            if not self.config.use_one_cycle_lr:
                # lr update
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch)
                    for param_group in self.optimizer.param_groups:
                        self.current_lr = param_group['lr']

            best = False
            diff = 0
            if self.monitor_mode != 'off':
                improved = (self.monitor_mode == 'min' and eval_log[
                    'val_' + self.monitor_metric] < self.monitor_best) or \
                           (self.monitor_mode == 'max' and eval_log['val_' + self.monitor_metric] > self.monitor_best)
                if improved:
                    self.monitor_best = eval_log['val_' + self.monitor_metric]
                    self.last_monitor_iou = self.monitor_iou
                    self.last_monitor_epoch = self.monitor_epoch
                    self.monitor_epoch = eval_log['epoch']
                    self.monitor_iou = eval_log['val_IoU']

                    best = True
                    self.best_epoch = eval_log['epoch']
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stop:
                    print("     + Validation Performance didn\'t improve for {} epochs."
                          "     + Training stop :/"
                          .format(self.not_improved_count))
                    break
            if epoch % self.save_period == 0 or best == True:
                self._save_ckpt(epoch, best=best)

        # save history file
        print("     + Saving History ... ... ")
        hist_path = os.path.join(self.log_dir, 'history1.txt')
        with open(hist_path, 'w') as f:
            f.write(str(self.history))

    def _train_epoch(self, epoch):
        ave_total_loss = AverageMeter()

        self.evaluator.reset()
        scaler = grad_scaler.GradScaler()
        if self.config.loss == 'crossentropy':
            L_seg = nn.CrossEntropyLoss(ignore_index=255)
        elif self.config.loss == 'bceloss':
            L_seg = nn.BCEWithLogitsLoss()

        # set model mode
        self.model.train()
        utils.util.set_seed(self.config.random_seed+epoch)
        for steps, (data, target, filename) in tqdm(enumerate(self.train_data_loader, start=1)):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            with autocast():
                output = self.model(data)
                if self.config.loss == 'bceloss':
                    logits = torch.squeeze(output["logits1"], 1)
                    probability = torch.sigmoid(logits)
                elif self.config.loss == 'crossentropy':
                    logits= output["logits1"]
                    target = target.long()
                loss = L_seg(logits, target)

            if self.config.use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.config.loss == 'bceloss':
                probability[probability < 0.5] = 0
                probability[probability >= 0.5] = 1
                pred = probability.view(-1).long()
            elif self.config.loss == 'crossentropy':
                pred = torch.argmax(logits, dim=1)
                pred = pred.view(-1).long()

            label = target.view(-1).long()

            self.evaluator.add_batch(label, pred)
            ave_total_loss.update(loss.item())

            if self.config.use_one_cycle_lr:
                # lr update
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    for param_group in self.optimizer.param_groups:
                        self.current_lr = param_group['lr']

        acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
        acc_class = self.evaluator.Pixel_Accuracy_Class().cpu().detach().numpy()
        miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
        confusion_matrix1 = self.evaluator.get_confusion_matrix().cpu().detach().numpy()
        TP, FP, FN, TN = self.evaluator.get_base_value()
        iou = self.evaluator.get_iou().cpu().detach().numpy()
        prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
        recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
        f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()

        #  train log
        self.history['train']['epoch'].append(epoch)
        self.history['train']['loss'].append(ave_total_loss.average())
        self.history['train']['acc'].append(acc.tolist())
        self.history['train']['iou'].append(iou[1])
        self.history['train']['prec'].append(prec[1])
        self.history['train']['recall'].append(recall[1])
        self.history['train']['f_score'].append(f1_score[1])

        return {
            'epoch': epoch,
            'loss': ave_total_loss.average(),
            'acc': acc,
            'iou': iou[1],
            'prec': prec[1],
            'recall': recall[1],
            'f_score': f1_score[1],
        }

    def _train_epoch_TFCSD(self, epoch):
        ave_total_loss = AverageMeter()

        self.evaluator.reset()
        scaler = grad_scaler.GradScaler()
        L_TEL = TreeEnergyLoss.TreeEnergyLoss(sigma=self.config.sigma)
        L_seg = nn.CrossEntropyLoss(ignore_index=255)
        T = self.config.temperature

        # set model mode
        self.model.train()
        utils.util.set_seed(self.config.random_seed + epoch)
        for steps, (data, target, filename) in tqdm(enumerate(self.train_data_loader, start=1)):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast():
                output = self.model(data)
                target = target.long()
                L_seg_value= L_seg(output["logits1"], target)

                unlabeled_RoIs = (target == 255)
                L_TEL_value= L_TEL(output["logits2"], data, output["embed_feat2"], unlabeled_RoIs, filename)
                L_con_value = self._loss_js_div(output["logits1"]/T, output["logits2"]/T)
                loss = L_seg_value + self.config.alpha * L_TEL_value + self.config.beta * L_con_value

            if self.config.use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            pred = torch.argmax(output["logits1"], dim=1)
            pred = pred.view(-1).long()

            label = target.view(-1).long()

            self.evaluator.add_batch(label, pred)
            ave_total_loss.update(loss.item())

            if self.config.use_one_cycle_lr:
                # lr update
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    for param_group in self.optimizer.param_groups:
                        self.current_lr = param_group['lr']

        acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
        acc_class = self.evaluator.Pixel_Accuracy_Class().cpu().detach().numpy()
        miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
        confusion_matrix1 = self.evaluator.get_confusion_matrix().cpu().detach().numpy()
        TP, FP, FN, TN = self.evaluator.get_base_value()
        iou = self.evaluator.get_iou().cpu().detach().numpy()
        prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
        recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
        f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()

        #  train log
        self.history['train']['epoch'].append(epoch)
        self.history['train']['loss'].append(ave_total_loss.average())
        self.history['train']['acc'].append(acc.tolist())
        self.history['train']['iou'].append(iou[1])
        self.history['train']['prec'].append(prec[1])
        self.history['train']['recall'].append(recall[1])
        self.history['train']['f_score'].append(f1_score[1])

        return {
            'epoch': epoch,
            'loss': ave_total_loss.average(),
            'acc': acc,
            'iou': iou[1],
            'prec': prec[1],
            'recall': recall[1],
            'f_score': f1_score[1],
        }

    def _eval_epoch(self, epoch):
        ave_total_loss = AverageMeter()
        self.evaluator.reset()
        if self.config.loss == 'crossentropy':
            L_seg = nn.CrossEntropyLoss(ignore_index=255)
        elif self.config.loss == 'bceloss':
            L_seg = nn.BCEWithLogitsLoss()
        self.model.eval()

        with torch.no_grad():
            for steps, (data, target, filename) in enumerate(self.valid_data_loder, start=1):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(data)

                if self.config.loss == 'bceloss':
                    logits = torch.squeeze(output["logits1"], 1)
                    probability = torch.sigmoid(logits)
                elif self.config.loss == 'crossentropy':
                    logits= output["logits1"]
                    target = target.long()

                loss = L_seg(logits, target)

                if self.config.loss == 'bceloss':
                    probability[probability < 0.5] = 0
                    probability[probability >= 0.5] = 1
                    pred = probability.view(-1).long()
                elif self.config.loss == 'crossentropy':
                    pred = torch.argmax(logits, dim=1)
                    pred = pred.view(-1).long()

                label = target.view(-1).long()

                self.evaluator.add_batch(label, pred)
                # update ave metrics
                ave_total_loss.update(loss.item())

            # calculate metrics
            acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
            acc_class = self.evaluator.Pixel_Accuracy_Class().cpu().detach().numpy()
            miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
            fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union().cpu().detach().numpy()
            confusion_matrix1 = self.evaluator.get_confusion_matrix().cpu().detach().numpy()
            TP, FP, FN, TN = self.evaluator.get_base_value()
            iou = self.evaluator.get_iou().cpu().detach().numpy()
            prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
            recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()
            kappa_coe = self.evaluator.Kapaa_coefficient().cpu().detach().numpy()

            print('Epoch {} validation done !'.format(epoch))
            print('lr: {:.8f}\n'
                  'IoU: {:6.4f},       Accuracy: {:6.4f},    Loss: {:.6f},\n'
                  'Precision: {:6.4f},  Recall: {:6.4f},      F_Score: {:6.4f}'
                  .format(self.current_lr,
                          iou[1], acc, ave_total_loss.average(),
                          prec[1], recall[1], f1_score[1]))

        self.history['valid']['epoch'].append(epoch)
        self.history['valid']['loss'].append(ave_total_loss.average())
        self.history['valid']['acc'].append(acc.tolist())
        self.history['valid']['iou'].append(iou[1])
        self.history['valid']['prec'].append(prec[1])
        self.history['valid']['recall'].append(recall[1])
        self.history['valid']['f_score'].append(f1_score[1])

        #  validation log
        return {
            'epoch': epoch,
            'val_Loss': ave_total_loss.average(),
            'val_Accuracy': acc,
            'val_IoU': iou[1],
            'val_Precision': prec[1],
            'val_Recall': recall[1],
            'val_F_score': f1_score[1],
        }

    def _resume_ckpt(self, resume_file):
        """
        :param resume_file: checkpoint file name
        :return:
        """
        resume_path = os.path.join(resume_file)
        print("     + Loading Checkpoint: {} ... ".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.monitor_best = 0.0
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("     + Model State Loaded ! :D ")

        print("     + Optimizer State Loaded ! :D ")
        print("     + Checkpoint file: '{}' , Start epoch {} Loaded !\n"
              "     + Prepare to run ! ! !"
              .format(resume_path, self.start_epoch))

    def _save_ckpt(self, epoch, best):
        # save model ckpt
        state = {
            'epoch': epoch,
            'arch': str(self.model),
            'history': self.history,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.monitor_best,
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-ep{}.pth'.format(epoch))
        best_filename = os.path.join(self.checkpoint_dir, 'checkpoint-best.pth')
        current_best_filename = os.path.join(self.checkpoint_dir,
                                             'checkpoint-ep{}-iou{:.4f}.pth'.format(epoch, self.monitor_iou))
        if best:
            # copy the last best model
            print("     + Saving Best Checkpoint : Epoch {}  path: {} ...  ".format(epoch, best_filename))
            torch.save(state, best_filename)
            shutil.copyfile(best_filename, current_best_filename)
        else:
            start_save_epochs = 1
            if epoch > start_save_epochs:
                print("     + After {} epochs, saving Checkpoint per {} epochs, path: {} ... ".format(start_save_epochs,
                                                                                                      self.save_period,
                                                                                                      filename))
                torch.save(state, filename)


