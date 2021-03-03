import sys
sys.path.insert(1, '/home/habib/myResearch/MONAI-FL/')

from communication.network import selectDevices

# Step 1: Server initiates the FL protocol
def startProtocol():
    dev = selectDevices()
    if(dev):
        print("Network is ready for training...")
    return print("Executing FL Protocol...")

# Step 2: Selects the devices and determine the number of workers from configuration file


# Step 3 on the Server
# Server will push the updated model to the selected devices → startTraining(n)
def pushModel(selectedDevices):
    establish_Connection_With_Selected_Devices_in_the_List ##input selectedDevices
    send_Model_to_selectedDevices # input selected devices
    return print("Model Bootstrapped...")

# Step 4 on the Server
# startTraining function will read the configuration file about clients (IP, port), model architecture, hyper-parameters, and which devices needs to perform the training and which will perform the validation on current model iteration. The function also detrmines the noise level for secure aggregation
def startTraining():
    Read_List_of_Devices
    Read_TrainingConfigurations
    beginTraining()
    return print("Executing FL Protocol...")


# Step 5 on the Workers
# Workers execute the training pipeline and save the weights in the local storage
def LocalTraining():
    print("Initiate and Execute complete local training pipeline")
    weights = 0.0
    return weights

# Step 6 on the Wokers
# Workers add the noise and return the parameters to the server
localWeights = LocalTraining
noise = 0
def addNoise(lw):
    noisyWeights = lw + noise
    return noisyWeights    

def transferWeights(server):
    return print("Transferring weights to the server: ", server)  


localWeights=0.0   
server = "127.0.0.1:8800"
secureWeights = addNoise(localWeights)
transferWeights(server) 

# Step 7 on the Server
# Server collects the noisy parameters, perform aggregations, cancels out the noise, retrain the model and stores it for next round of iterations. 

def collectParameters():
    print("Collecting paramters...")
    params = 100
    return params

def secureAggregation(secParameters):
    print("Aggregating parameters securely...")
    secparms = 200
    return secparms

def cancelNoise(secAggregatedParameters):
    print("Cancelling Noise from aggregated parameters...")
    noiseFree = 15
    return noiseFree

def reTrain(modelParameters):
    print("retraing model")
    print("saving model for next iteration")

secureParamList = collectParameters() # params from all the devices
secureAggregatedParameters = secureAggregation(secureParamList)
noiseFreeParameters = cancelNoise(secureAggregatedParameters)
newModel = reTrain(noiseFreeParameters)





