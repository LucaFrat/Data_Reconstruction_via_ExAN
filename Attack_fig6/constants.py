""" HyperParameters """

import torch

N_EPOCHS = 1
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_TEST = 10
OUT_SIZE = 5
HDIM = 512
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 1000
P_DROPOUT = 0.3
TOL = 0.00000001

RANDOM_SEED = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHOW = True
