# This file contains functions to save and load the model checkpoints in the local storage. It also interfaces with client_trainer and client_communicator
import sys
import os

ProjecttDir = os.getcwd()
sys.path.insert(1, ProjecttDir)

#path for linux distribution
#sys.path.insert(1, '/home/habib/myResearch/MONAI-FL')
#path for windows installation
#sys.path.insert(1, 'C:/Users/mhreh/research/MONAI-FL/MONAI-FL/')
import torch
from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar
from monai.networks.nets import densenet121


savedir = 'client_model'
checkpointdir = os.path.join('./checkpoints', savedir)
FILE = os.path.join(checkpointdir, 'checkpoint.pth.tar')
 
# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
modelCheckPoint = {
    'epoch': 0,
    'state_dict': {},
    'optimizer': {},
    'best_metric': 0
    }

def getModel(argsModel):
  if argsModel == 'desnsenet':
    net_glob = densenet121(spatial_dims=2, in_channels=1, out_channels=num_class)
  else:
    exit('Error: unrecognized model')
  #print(net_glob)
  #net_glob.train()
  # copy weights
  w_glob = net_glob.state_dict()
  return net_glob

def modelBootstrap():
  #colecting model from server storage and sending it to devices in the list.
  #path for linux distribution
  #FILE = '/home/habib/myResearch/MONAI-FL/save/models/client/testmodel.pth'
  #path for windows installation
#  FILE = 'C:/Users/mhreh/research/MONAI-FL/MONAI-FL/save/models/server/testmodel.pth'
  
  model = getModel(args.model)
  try:
    modelCheckPoint = torch.load(fullpath)
  except FileNotFoundError:
    print("client has no model to bootstrap with!")
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    modelCheckPoint = {
                    'epoch': best_metric_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_metric': best_metric
                    }

    torch.save(modelCheckPoint, fullpath)
  #model.eval() to be executed when need to update the model at server or client
  if model:
    modelCheckPoint = torch.load(fullpath)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    model.load_state_dict(modelCheckPoint['state_dict'])
    optimizer.load_state_dict(modelCheckPoint['optimizer'])
    model.eval()
    # - or -
    # model.train()
    #print(model)
    torch.save(modelCheckPoint, fullpath)
    print("sending model")
    model = False
    return modelCheckPoint

