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









def flProtocol():
  modelBoostrap()
  setModelConfigurations()
  getModelUpdate()
  secureAggregation()
  setLocalModelUpdate()
  checkPoint(state, filename)
  return print("Executing FL Protocol...")

flProtocol()

def modelBoostrap():
  #colecting model from server storage and sending it to devices in the list.
  print("Bootstrap model")
def setModelConfigurations():
  #colecting model configurations from server storage and sending it to devices in the list.
  print("send model configurations")
  
def getModelUpdate():
  # receiving model gradients from all FL participants
  print("Received model weights from devices")
  
def secureAggregation():
  # calling and using the federated aggregation algorithms
  print("Perform secure Aggregation")
  
def setLocalModelUpdate():
  # retrain the model on aggregated parameters
  print("Update Local model")
  
def checkPoint(state, filename="modelname.pth.tar"):
  print("saving checkpoint...") # save next model checkpoint in server
  torch.save(state, filename)

FILE = "model.pth"
torch.save(model.state_dict(), FILE)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(FILE))
model.eval()

checkPoint = {
  'state_dict': model.state_dict(), # saves model weights and biases
  'optimizer': optimizer.state_dict() # save optimizer hyperparameters
  }
save_checkpoint(checkPoint)


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





