import sys
#path for linux distribution
# sys.path.insert(1, '/home/habib/myResearch/MONAI-FL')
#path for windows installation
sys.path.insert(1, 'C:/Users/mhreh/research/MONAI-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.options import args_parser

from models.Nets import MLP, CNNMnist, CNNCifar
#from communication.network import getNetworkConfigurations
from communication.server import start  
# Step 1: Server initiates the FL protocol

# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

dataset_train = []
dataset_test = []
img_size = []

modelCheckPoint = {
    "epoch": 0,
    "model_state": {},  
    "optim_state": {}
    }


