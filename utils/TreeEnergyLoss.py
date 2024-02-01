import os
import sys
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from .lib_tree_filter.modules.tree_filter import TreeFilter2D

class TreeEnergyLoss(nn.Module):
    def __init__(self, sigma=0.002, configer=None):
        super(TreeEnergyLoss, self).__init__()
        self.configer = configer

        self.weight = 1.0
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=sigma)

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs,filenames=None):
        with torch.no_grad():
            batch, _, h, w = preds.size()
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            N = unlabeled_ROIs.sum()

        if N > 0:
            prob = torch.softmax(preds, dim=1)

            # low-level MST
            tree = self.mst_layers(low_feats)
            AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree, filenames=filenames)

            # high-level MST
            if high_feats is not None:
                tree = self.mst_layers(high_feats)
                AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False, filenames=filenames)

            tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()
            if N > 0:
                tree_loss /= N
        else:
            tree_loss = torch.tensor([0.0], device=preds.device)
        return self.weight * tree_loss



