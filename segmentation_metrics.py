"""Common image segmentation metrics.
''' Git : https://github.com/kevinzakka/pytorch-goodies/blob/master/metrics.py
"""

import torch
import numpy as np


EPS = 1e-10

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])

def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = np.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).astype(float)
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    EPS = 1e-10
    correct = np.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.
    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.
    Args:
        hist: confusion matrix.
    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    EPS = 1e-10
    correct_per_class = np.diag(hist)
    total_per_class = hist.sum(axis=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = np.nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    EPS = 1e-10
    A_inter_B = np.diag(hist)
    A = hist.sum(axis=1)
    B = hist.sum(axis=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = np.nanmean(jaccard)
    return avg_jacc

def jaccard_index_m(y_true, y_pred):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    EPS = 1e-10
    intersection = np.sum(np.multiply(y_pred, y_true))
    A_sum = np.sum(np.multiply(y_pred,y_pred))
    B_sum = np.sum(np.multiply(y_true,y_true))
    jaccard = intersection / (A_sum + B_sum - intersection + EPS)
    avg_jacc = np.nanmean(jaccard)
    return avg_jacc

def iou(y_true, y_pred):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    intersection = np.logical_and(y_pred, y_true)
    union = np.logical_or(y_pred, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    EPS = 1e-10
    A_inter_B = np.diag(hist)
    A = hist.sum(axis=1)
    B = hist.sum(axis=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = np.nanmean(dice)
    return avg_dice


def eval_metrics(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.
    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    hist = np.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count