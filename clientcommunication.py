# This file will store all the communication related functions from the client end. This will be directly communicating with server

import sys
#path for linux distribution
#sys.path.insert(1, '/home/habib/myResearch/MONAI-FL')
#path for windows installation
sys.path.insert(1, 'C:/Users/mhreh/research/MONAI-FL/MONAI-FL/')

import socket
import torch
import os
import tqdm
import time
import pickle
import json
import subprocess

from clientfilehandler import modelBootstrap


HEADER  = 64
PORT = 8000
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
CONNECT_MESSAGE = "Connected"

MODEL_MESSAGE = "model"
WEIGHTS_MESSAGE = "weights"
PARAMETERS_MESSAGE = "parameters"
CONFIGURATION_MESSAGE = "configurations"

# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

SERVER = "127.0.0.1"  
ADDR = (SERVER,PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

#path for linux distribution
#FILE = '/home/habib/myResearch/MONAI-FL/save/models/server/testmodel.pth'
#path for windows installation
FILE = 'C:/Users/mhreh/research/MONAI-FL/MONAI-FL/save/models/server/testmodel.pth'

modelCheckPoint = {
    "epoch": 0,
    "model_state": {},  
    "optim_state": {}
    }

def recvall(n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = client.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def sendMessage(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    client.send(send_length)
    client.send(message)

def receiveMessage():
    msg =''
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
    return msg

def sendModel():
    # the name of file we want to send, make sure it exists
    filename = "testmodel.pth"
    # get the file size
    filesize = os.path.getsize(FILE)
    # send the filename and filesize
    client.send(f"{filename}{SEPARATOR}{filesize}".encode())

    # start sending the file
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(FILE, "rb") as f:
        while True:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
            # file transmitting is done
                break
            # we use sendall to assure transimission in 
            # busy networks
            client.sendall(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))
            #print(bytes_read)
    client.shutdown(socket.SHUT_WR)
    #close the socket
    #conn.close()

def sendWeights(msgData):
    #message = (msgData).encode(FORMAT)
    #message = json.dumps(msgData)
    message = pickle.dumps(msgData)
    #print(message)
    msg_length = len(message)
    #print(msg_length)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    conn.send(send_length)
    conn.sendall(message)

def receiveModel():
    received = client.recv(BUFFER_SIZE).decode()
    filename, filesize = received.split(SEPARATOR)
    # remove absolute path if there is
    FILE = os.path.basename(filename)
    # convert to integer
    filesize = int(filesize)

    # start receiving the file from the socket
    # and writing to the file stream
    progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(FILE, "wb") as f:
        while True:
            # read 1024 bytes from the socket (receive)
            bytes_read = client.recv(BUFFER_SIZE)
            time.sleep(0.1)
            if not bytes_read:    
                print("No More Data :-)")
                # file transmitting is done
                break
            # write to the file the bytes we just received
            #print(bytes_read)
            f.write(bytes_read)
            # update the progress bar
            #print(bytes_read)
            progress.update(len(bytes_read))

def receiveWeights():
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        mssg = recvall(msg_length)#.decode(FORMAT)
        #print(mssg)
        #print(msg_length)
        #print(len(mssg))
        msssg = pickle.loads(mssg)
    return msssg

def handle_server():
    print(CONNECT_MESSAGE)
    sendMessage(CONNECT_MESSAGE) # 1
    server_message  = receiveMessage() # 2
    print(server_message)
    print("Starting FL protocol at client")
    glob_epoch = int (receiveMessage()) # 3
    loc_epoch = 0
    while loc_epoch < glob_epoch:
        print("Local Epoch: "+ str(loc_epoch+1) + "/" + str(glob_epoch))
        if loc_epoch == 0:
            print ("This is first round")
            receiveModel()
            print("Initial Global Model Transferred!")
            #call(["Python", {client_trainer.py}])
            subprocess.run('python monai-fl-example-client.py')
        else:
            print ("This is round: ", str(loc_epoch+1))
        
        
        
        loc_epoch+=1
    
    server_message  = receiveMessage()
    print(server_message)
    #model = modelBootstrap()
    # print("Global Epoch: " + glob_epoch)
    # if glob_epoch == 0:
    #     print("We are going to begin training for " + str(GlobalEpochs) + " rounds")
    #     print("Acquiring latest model")
    #     receiveModel() # 4
    #     os.system('python client_trainer.py')
        #ExecuteLocalPipeline(local_epoch) # it will execute the DL algorithms
        #modelCP = torch.load(FILE)
        #print(modelCP['model_state'])
        #sendWeights(modelCP['model_state'])

 #   else:
        #connected = True
        #while connected:
        #    print("Global Epoch: ", glob_epoch)
            # #sendMessage(WEIGHTS_MESSAGE)
            # #print(receiveMessage())
            # Global_Weights = receiveWeights()
            # modelCP = torch.load(FILE)
            # optimizer = torch.optim.SGD(model.parameters(), lr=0)
            # model.load_state_dict(Global_Weights)
            # optimizer.load_state_dict(modelCP['optim_state'])
            # model.eval()
            # #modelCP['model_state'] = Global_Weights
            # model.save(modelCP, FILE)
            # #ExecuteLocalPipeline(local_epoch)
            # modelCP = torch.load(FILE)
            # sendWeights(modelCP['model_state'])
            # statusMessage = receiveMessage()
            # if statusMessage == DISCONNECT_MESSAGE:
            #connected = False
        
        
handle_server()

