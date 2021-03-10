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
  FILE = 'C:/Users/mhreh/research/MONAI-FL/save/models/server/testmodel.pth'
  
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
  
def getModelUpdate():
  # receiving model gradients from all FL participants
  print("Received model weights from devices")
  
def secureAggregation():
  # calling and using the federated aggregation algorithms
  print("Perform secure Aggregation")
  
def setLocalModelUpdate():
  # retrain the model on aggregated parameters
  print("Update Local model")
  
def flProtocol():
  modelBoostrap()
  #getModelUpdate()
  #secureAggregation()
  #setLocalModelUpdate()
  #checkPoint(state, filename)
  return print("Executing FL Protocol...")

flProtocol()

 #   dev = getNetworkConfigurations()
 #   print(dev)
  #  serv = tuple(dev[0])
  #  print(len(dev))
  #  if(dev):
 #       print("Network is ready for training...")
 #       print(tuple(dev[0]))
  #      ip = dev[0][0]
  #      port = dev[0][1]
  #      serv = (tuple(ip), tuple(port))
     #   print(ip, port)
     #   serv = (ip, port)
 #       start(serv)
 #       print(dev[1])
 #   return print("Executing FL Protocol...")
 





#   send_Model_to_selectedDevices(dev)
# Step 2: Selects the devices and determine the number of workers from configuration file

# Step 3 on the Server
# Server will push the updated model to the selected devices → startTraining(n)
# def pushModel(selectedDevices):
#     while selectedDevices:
#         sendmodel(slectedDevices.getitem()) # input selected devices
#     return print("Model Bootstrapped...")

# # Step 4 on the Server
# # startTraining function will read the configuration file about clients (IP, port), model architecture, hyper-parameters, and which devices needs to perform the training and which will perform the validation on current model iteration. The function also detrmines the noise level for secure aggregation
# def startTraining():
#     Read_List_of_Devices
#     Read_TrainingConfigurations
#     beginTraining()
#     return print("Executing FL Protocol...")


# # Step 5 on the Workers
# # Workers execute the training pipeline and save the weights in the local storage
# def LocalTraining():
#     print("Initiate and Execute complete local training pipeline")
#     weights = 0.0
#     return weights

# # Step 6 on the Wokers
# # Workers add the noise and return the parameters to the server
# localWeights = LocalTraining
# noise = 0
# def addNoise(lw):
#     noisyWeights = lw + noise
#     return noisyWeights    

# def transferWeights(server):
#     return print("Transferring weights to the server: ", server)  


# localWeights=0.0   
# server = "127.0.0.1:8800"
# secureWeights = addNoise(localWeights)
# transferWeights(server) 

# # Step 7 on the Server
# # Server collects the noisy parameters, perform aggregations, cancels out the noise, retrain the model and stores it for next round of iterations. 

# def collectParameters():
#     print("Collecting paramters...")
#     params = 100
#     return params

# def secureAggregation(secParameters):
#     print("Aggregating parameters securely...")
#     secparms = 200
#     return secparms

# def cancelNoise(secAggregatedParameters):
#     print("Cancelling Noise from aggregated parameters...")
#     noiseFree = 15
#     return noiseFree

# def reTrain(modelParameters):
#     print("retraing model")
#     print("saving model for next iteration")

# secureParamList = collectParameters() # params from all the devices
# secureAggregatedParameters = secureAggregation(secureParamList)
# noiseFreeParameters = cancelNoise(secureAggregatedParameters)
# newModel = reTrain(noiseFreeParameters)





