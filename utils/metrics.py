import numpy as np
import torch

class Evaluator_tensor(object):
    """
    evaluation metrics for semantic segmentation
    """

    def __init__(self, num_class, device):
        self.device = device
        self.num_class = num_class
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).long().to(self.device)
        self.iou = torch.zeros(self.num_class).to(self.device)
        self.accuracy_class = torch.zeros(self.num_class).to(self.device)
        self.precision_class = torch.zeros(self.num_class).to(self.device)
        self.recall_class = torch.zeros(self.num_class).to(self.device)
        self.f1_score_class = torch.zeros(self.num_class).to(self.device)
        self.TP = torch.zeros(self.num_class).to(self.device)
        self.FP = torch.zeros(self.num_class).to(self.device)
        self.FN = torch.zeros(self.num_class).to(self.device)
        self.TN = torch.zeros(self.num_class).to(self.device)

    def Pixel_Accuracy(self):
        """
        calculate pixel accuracy
        """
        Acc = torch.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum()+1e-8)
        return Acc

    def Pixel_Accuracy_Class(self):
        """
        Calculate per class pixel accuracy
        """
        Acc = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=1)+1e-8)
        self.accuracy_class = Acc
        Acc = torch.mean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        """
        calculate mean intersection over union
        """
        MIoU = torch.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) -
                torch.diag(self.confusion_matrix)+1e-8)
        self.iou = MIoU
        MIoU = torch.mean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        calculate frequency weighted intersection over union
        """
        freq = torch.sum(self.confusion_matrix, dim=1) / (torch.sum(self.confusion_matrix)+1e-8)
        iu = torch.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) -
                torch.diag(self.confusion_matrix)+1e-8)

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        """
        generate confusion matrix based on ground truth and prediction
        """
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask] + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """
        add a batch of image to the confusion matrix
        """
        assert gt_image.shape == pre_image.shape
        tem_cm = self.confusion_matrix.clone().detach()
        self.confusion_matrix = tem_cm + self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """
        reset confusion matrix to zero
        """
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).long().to(self.device)

    def get_confusion_matrix(self):
        """
        get confusion matrix
        """
        return self.confusion_matrix

    def get_base_value(self):
        """
        get TP, FP, FN, TN
        """
        self.FP = self.confusion_matrix.sum(dim=0) - torch.diag(self.confusion_matrix)
        self.FN = self.confusion_matrix.sum(dim=1) - torch.diag(self.confusion_matrix)
        self.TP = torch.diag(self.confusion_matrix)
        self.TN = self.confusion_matrix.sum() - (self.FP + self.FN + self.TP)
        return self.TP, self.FP, self.FN, self.TN

    def get_iou(self):
        return self.iou

    def Pixel_Precision_Class(self):
        self.precision_class = self.TP / (self.TP + self.FP + 1e-8)
        return self.precision_class

    def Pixel_Recall_Class(self):
        self.recall_class = self.TP / (self.TP + self.FN + 1e-8)
        return self.recall_class

    def Pixel_F1_score_Class(self):
        self.f1_score_class = 2 * self.TP / (2 * self.TP + self.FP + self.FN + 1e-8)
        return self.f1_score_class

    def Kapaa_coefficient(self):
        """
        calculate kappa coefficient
        """
        cm = self.confusion_matrix
        po = cm.diagonal().sum() / (cm.sum()+ 1e-8)

        sum1 = 0
        for i in range(cm.shape[0]):
            sum1 += cm[i, :].sum() * cm[:, i].sum()
        pe = sum1 / (cm.sum() * cm.sum()+ 1e-8)
        return (po - pe) / (1 - pe + 1e-8)

    def Kapaa_coefficient_sklearn(self):
        """
        calculate kappa coefficient (sklearn algorithm)
        """
        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        sum0 = torch.sum(cm, dim=0)
        sum1 = torch.sum(cm, dim=1)
        expected = torch.outer(sum0, sum1) / (torch.sum(sum0)+ 1e-8)

        w_mat = torch.ones([n_classes, n_classes], dtype=torch.int)
        w_mat.flat[:: n_classes + 1] = 0

        k = torch.sum(w_mat * cm) / (torch.sum(w_mat * expected) + 1e-8)
        return 1 - k


if __name__ == '__main__':
    evaluator = Evaluator_tensor(2, device='cpu')
    a = torch.tensor(np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0]))
    b = torch.tensor(np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1]))
    evaluator.add_batch(a, b)

    acc = evaluator.Pixel_Accuracy()
    acc_class = evaluator.Pixel_Accuracy_Class()
    miou = evaluator.Mean_Intersection_over_Union()
    fwiou = evaluator.Frequency_Weighted_Intersection_over_Union()
    confusion_matrix1 = evaluator.get_confusion_matrix()
    TP, FP, FN, TN = evaluator.get_base_value()
    iou = evaluator.get_iou()
    prec = evaluator.Pixel_Precision_Class()
    recall = evaluator.Pixel_Recall_Class()
    f1_score = evaluator.Pixel_F1_score_Class()
    kappa_coe = evaluator.Kapaa_coefficient()
    print('Class:    ', 2, ' Average')
    np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
    print('IoU:      ', iou)
    print('Precision:', prec)
    print('Recall:   ', recall)
    print('F_Score:  ', f1_score)
    np.set_printoptions(formatter={'int': '{:14}'.format})
    print('Confusion_matrix:')
    print(confusion_matrix1)
    print('Kappa_Coefficient:{:10.6f}'.format(kappa_coe))
