import sys
sys.path.insert('..')

import argparse
from utils import utils
import numpy as np
import torch

def img2np(img):
    img = np.array(img, dtype=np.int32)
    img = np.expand_dims(img, axis=-1)
    return img

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_arugment('-d', '--dataset', default='MNIST')
    parser.add_argument('-b', '--batch_size', default=128)

    return parser.parse_args()

def main():
    args = parse_args()
    (train_set, val_set, test_set), (train_loader, val_loader, test_loader) = \
            utils.load_data(args.dataset, img2np, [50000, 10000], args.batch_size)



    
