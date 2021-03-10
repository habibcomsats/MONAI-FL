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
#from communication.server import start  
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

argsDataset = args.dataset
def getDataset(argsDataset):
  # load dataset and split users
  if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    # if args.iid:                                           # sample users
    #   dict_users = mnist_iid(dataset_train, args.num_users)
    # else:
    #   dict_users = mnist_noniid(dataset_train, args.num_users)
  elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
      # if args.iid:
      #   dict_users = cifar_iid(dataset_train, args.num_users)
      # else:
      #   exit('Error: only consider IID setting in CIFAR10')
  else:
    exit('Error: unrecognized dataset')
  img_size = dataset_train[0][0].shape

def getModel(argsModel):
  # build model
  if argsModel == 'cnn' and args.dataset == 'cifar':
    net_glob = CNNCifar(args=args).to(args.device)
  elif argsModel == 'cnn' and args.dataset == 'mnist':
    net_glob = CNNMnist(args=args).to(args.device)
  elif argsModel == 'mlp':
    print(img_size)
    len_in = 1
    for x in img_size:
      len_in *= x
    net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
  else:
    exit('Error: unrecognized model')
  #print(net_glob)
  #net_glob.train()

  # copy weights
  w_glob = net_glob.state_dict()

  return net_glob

def modelBoostrap():
  #colecting model from server storage and sending it to devices in the list.
  FILE = 'C:/Users/mhreh/research/MONAI-FL/save/models/client/testmodel.pth'
  
  model = getModel(args.model)
  try:
    modelCheckPoint = torch.load(FILE)
    
  except FileNotFoundError:
    print("Model.pth not found")
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    modelCheckPoint = {
    "epoch": 90,
    "model_state": model.state_dict(),  
    "optim_state": optimizer.state_dict()
    }
    torch.save(modelCheckPoint, FILE)
    #print(modelCheckPoint)
    print("local model ready for sending...")
    model = False
  
  
  #model.eval() to be executed when need to update the model at server or client
  if model:
    modelCheckPoint = torch.load(FILE)
    epoch = modelCheckPoint['epoch']
    print(epoch)
    modelState = modelCheckPoint['model_state'] #model.load_state_dict(model['model_state'])
    print(modelState)
    optimizerState = (modelCheckPoint['optim_state'])
    print(optimizerState)
   # optimizer = torch.optim.SGD(model.parameters(), lr=0)
    model.eval()
    # - or -
    # model.train()
    #print(model)
    torch.save(modelCheckPoint, FILE)
    #sendModel(FILE)
    print("sending model")
    model = False
 # else:
   # modelCheckPoint = excuteServerPipeline()
   # sendModel(modelCheckPoint)
    # torch.save(modelCheckPoint, FILE)
    # print(modelCheckPoint)
    # print("local model ready yp send")
  return modelCheckPoint
