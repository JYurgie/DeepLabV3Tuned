import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, multilabel_confusion_matrix, jaccard_score, plot_confusion_matrix
from model import createDeepLabv3
from trainer import train_model
import datahandler
import argparse
import os
import torch
from segmentation_losses import diceLoss, dice_coeffcient, DiceLossV2, BinaryDiceLoss, dice_loss, dice_loss_v3
from losses import DiceLoss
from segmentation_metrics import iou
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_directory", help='Specify the dataset directory path')
    parser.add_argument(
        "exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batchsize", default=4, type=int)

    args = parser.parse_args()


    bpath = args.exp_directory
    data_dir = args.data_directory
    epochs = args.epochs
    batchsize = args.batchsize
    # Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3()
    model.train()
    # Create the experiment directory if not present
    if not os.path.isdir(bpath):
        os.mkdir(bpath)


    # Specify the loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    #criterion = diceLoss()


    # Possible Losses ** Experimental **
    #criterion = torch.nn.BCEWithLogitsLoss(reduction='mean') # Probably not the right loss funciton for me
    #criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # LOSS DECREASED (SMALL DATASET) NO APPRECIABLE MASK
    #criterion = dice_loss_v3()
    #criterion = DiceLossV2()
    #criterion = BinaryDiceLoss()
    #criterion = DiceLoss()

    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evalutation metrics
    # metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
    # metrics = {'dice_coeff': dice_coeffcient, 'overall_acc': me.overall_pixel_accuracy, 'avg_per_class_acc': me.per_class_pixel_accuracy,
    #          'avg_jacc': me.jaccard_index, 'avg_dice': me.dice_coefficient}
    metrics = {'dice_coeff': dice_coeffcient, 'avg_jacc_m': jaccard_score, 'f1_score': f1_score, 'iou': iou, 'auroc': roc_auc_score}


    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(data_dir, batch_size=batchsize)
    trained_model = train_model(model, criterion, dataloaders, optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


    # Save the trained model
    # torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
    torch.save(model, os.path.join(bpath, 'weights.pt'))
