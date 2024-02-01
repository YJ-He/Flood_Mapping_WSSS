import torch
import numpy as np
import os
import time
import torch.nn as nn
from tqdm import tqdm
from utils.util import AverageMeter, ensure_dir
import shutil
from PIL import Image
from utils.metrics import Evaluator_tensor

class Tester(object):
    def __init__(self,
                 model,
                 config,
                 args,
                 test_data_loader,
                 class_name,
                 begin_time,
                 resume_file
                 ):

        # for general
        self.config = config
        self.args = args
        self.device = torch.device('cpu') if self.args.gpu == -1 else torch.device('cuda:{}'.format(self.args.gpu))
        self.class_name = class_name
        # for Test
        if isinstance(model, list):
            self.model = []
            for m in model:
                m = m.to(self.device)
                self.model.append(m)
        else:
            self.model = model.to(self.device)

        self.models = []

        # for time
        self.begin_time = begin_time

        # for data
        self.test_data_loader = test_data_loader

        # for resume/save path
        self.history = {
            "eval": {
                "loss": [],
                "acc": [],
                "iou": [],
                "time": [],
                "prec": [],
                "recall": [],
                "f_score": [],
            },
        }

        self.model_name = self.config.model_name
        self.log_dir = os.path.join(self.args.output, self.model_name,
                                    self.begin_time, 'log')

        if not self.args.only_prediction:
            self.test_log_path = os.path.join(self.args.output, 'test', 'log', self.model_name,
                                              self.begin_time)
            ensure_dir(self.test_log_path)

        self.predict_path = os.path.join(self.args.output, 'test', 'predict', self.model_name,
                                         self.begin_time)
        ensure_dir(self.predict_path)
        self.resume_ckpt_path = resume_file if resume_file is not None else \
            os.path.join(self.config.save_dir, self.model_name,
                         self.begin_time, 'checkpoint-best.pth')

        with open(os.path.join(self.predict_path, 'checkpoint.txt'), 'w') as f:
            f.write(self.resume_ckpt_path)
            self.model_name

        self.evaluator = Evaluator_tensor(self.config.nb_classes, self.device)

    def eval_and_predict(self):
        self._resume_ckpt()
        self.evaluator.reset()
        if self.config.loss == 'crossentropy':
            L_seg = nn.CrossEntropyLoss(ignore_index=255)
        elif self.config.loss == 'bceloss':
            L_seg = nn.BCEWithLogitsLoss()
        ave_total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            tic = time.time()
            for steps, (images, target, filenames) in tqdm(enumerate(self.test_data_loader, start=1)):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(images)
                if self.config.loss == 'bceloss':
                    logits = torch.squeeze(output["logits1"], 1)
                    probability = torch.sigmoid(logits)
                elif self.config.loss == 'crossentropy':
                    logits = output["logits1"]

                    target = target.long()

                loss = L_seg(logits, target)

                if self.config.loss == 'bceloss':
                    probability[probability < 0.5] = 0
                    probability[probability >= 0.5] = 1
                    self._save_pred(probability, filenames)
                    pred = probability.view(-1).long()
                elif self.config.loss == 'crossentropy':
                    pred = torch.argmax(logits, dim=1)
                    self._save_pred(pred, filenames)
                    pred = pred.view(-1).long()

                label = target.view(-1).long()

                self.evaluator.add_batch(label, pred)

                ave_total_loss.update(loss.item())
            total_time = time.time() - tic
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

            # display evaluation result at the end
            print('Evaluation phase !\n'
                  'Accuracy: {:6.4f}, Loss: {:.6f}'.format(
                acc, ave_total_loss.average()))
            np.set_printoptions(formatter={'int': '{: 9}'.format})
            print('Class:    ', self.class_name, ' Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            print('IoU:      ', np.hstack((iou, np.average(iou))))
            print('Precision:', np.hstack((prec, np.average(prec))))
            print('Recall:   ', np.hstack((recall, np.average(recall))))
            print('F_Score:  ', np.hstack((f1_score, np.average(f1_score))))
            np.set_printoptions(formatter={'int': '{:14}'.format})
            print('Confusion_matrix:')
            print(confusion_matrix1)

            # normalized confusion matrix
            np.set_printoptions(formatter={'float': '{: 7.4f}'.format})
            confusion_matrix_norm = confusion_matrix1 / np.sum(confusion_matrix1)
            print('Normalized_confusion_matrix:')
            print(confusion_matrix_norm)

            print('Kappa_Coefficient:{:10.6f}'.format(kappa_coe))
            print('Prediction Phase !\n'
                  'Total Time cost: {:.2f}s\n'
                  .format(total_time,
                          ))
        self.history["eval"]["loss"].append(ave_total_loss.average())
        self.history["eval"]["acc"].append(acc.tolist())
        self.history["eval"]["iou"].append(iou.tolist())
        self.history["eval"]["time"].append(total_time)

        self.history["eval"]["prec"].append(prec.tolist())
        self.history["eval"]["recall"].append(recall.tolist())
        self.history["eval"]["f_score"].append(f1_score.tolist())

        # save results to log file
        print("     + Saved history of evaluation phase !")
        hist_path = os.path.join(self.test_log_path, "history1.txt")
        with open(hist_path, 'w') as f:
            f.write(str(self.history).replace("'", '"'))
            f.write('\nKappa_Coefficient:{:10.6f}'.format(kappa_coe))
            f.write('\nConfusion_matrix:\n')
            f.write(str(confusion_matrix1))
            np.set_printoptions(formatter={'float': '{: 6.3f}'.format})
            f.write('\n Normalized_confusion_matrix:\n')
            f.write(str(confusion_matrix_norm))

            np.set_printoptions(formatter={'int': '{: 9}'.format})
            f.write('\nClass:    ' + str(self.class_name) + '  Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            format_iou = np.hstack((iou, np.average(iou)))
            format_prec = np.hstack((prec, np.average(prec)))
            format_recall = np.hstack((recall, np.average(recall)))
            format_f1_score = np.hstack((f1_score, np.average(f1_score)))
            f.write('\nIoU:      ' + str(format_iou))
            f.write('\nPrecision:' + str(format_prec))
            f.write('\nRecall:   ' + str(format_recall))
            f.write('\nF1_score: ' + str(format_f1_score))

        test_log_path1 = os.path.join(self.args.output, 'test', 'log', self.model_name, "history1.txt")
        if os.path.exists(test_log_path1):
            os.remove(test_log_path1)
        shutil.copy(hist_path, test_log_path1)
        if not self.args.is_test:
            hist_test_log_path = os.path.join(self.log_dir, "history1-test.txt")
            shutil.copy(hist_path, hist_test_log_path)
        else:
            input_dir_path=os.path.dirname(self.resume_ckpt_path)
            input_file_name=os.path.basename(self.resume_ckpt_path)
            output_dir=os.path.join(input_dir_path, 'batch_test')
            ensure_dir(output_dir)
            output_file_path=os.path.join(output_dir, input_file_name+'.txt')
            shutil.copy(hist_path, output_file_path)

        return iou[1], ave_total_loss.average()

    def predict(self):

        self._resume_ckpt()
        self.evaluator.reset()
        self.model.eval()
        with torch.no_grad():
            for steps, (images, filenames) in tqdm(enumerate(self.test_data_loader, start=1)):
                # images
                images = images.to(self.device, non_blocking=True)
                output = self.model(images)

                pred = torch.argmax(output["logits1"], dim=1)
                self._save_pred(pred, filenames)
            print("Predicting and Saving Done!\n")


    def _resume_ckpt(self):
        print("     + Loading ckpt path : {} ...".format(self.resume_ckpt_path))
        checkpoint = torch.load(self.resume_ckpt_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("     + Model State Loaded ! :D ")
        print("     + Checkpoint file: '{}' , Loaded ! \n"
              "     + Prepare to test ! ! !"
              .format(self.resume_ckpt_path))

    def _save_pred(self, binary_map, filenames):
        """
        save binary_map
        """
        for index, map in enumerate(binary_map):
            map = np.asarray(map.cpu(), dtype=np.uint8) * 255
            map = Image.fromarray(map)
            filename = filenames[index].split('\\')[-1].split('.')
            save_filename = filename[0] + '_binary.tif'
            save_path = os.path.join(self.predict_path, save_filename)
            map.save(save_path, compression='tiff_lzw')


