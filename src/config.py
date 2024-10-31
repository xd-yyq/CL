import os
import torch

class Config:
    def __init__(self):
        self.DATA_DIR = './data'
        self.DATASET_TYPE = 'cifar10'  # 在这里设置数据集类型，例如 'cifar10' 或 'cifar100'

        self.OUTPUT_DIR = './output'

        self.NUM_TASKS = 5
        self.NUM_EPOCHS = 1
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 16
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
