# This file will store all the communication related functions from the server end. This will be directly communicating with client

import sys
import os
ProjecttDir = os.getcwd()
sys.path.insert(1, ProjecttDir)

#path for linux distribution
#sys.path.insert(1, '/home/habib/myResearch/MONAI-FL')

#path for windows installation
#sys.path.insert(1, 'C:/Users/mhreh/research/MONAI-FL/MONAI-FL/')

import socket
import threading
import pickle
import torch
import tqdm
import os
import time
import json

import subprocess
import py_compile
from subprocess import Popen

import shutil

from serverfilehandler import modelBootstrap
from communication.network import getNetworkConfigurations
from utils.options import args_parser
#from networks.nets.densenet import *

HEADER = 64
#PORT = 8500

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step
fl_participants = getNetworkConfigurations()

SERVER =  fl_participants[0][0] # "127.0.1.1" #socket.gethostbyname(socket.gethostname())
print("Server's IP address is: "+ SERVER)
PORT = fl_participants[0][1]
print(PORT)
ADDR  = (SERVER, int(PORT)) 

FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "Disconnect!"
CONNECT_MESSAGE = "Connected"

MODEL_MESSAGE = "model"
WEIGHTS_MESSAGE = "weights"
PARAMETERS_MESSAGE = "parameters"
CONFIGURATION_MESSAGE = "configurations"

#path for linux distribution
#FILE = '/home/habib/myResearch/MONAI-FL/save/models/server/testmodel.pth'
#path for windows installation
#FILE = 'C:/Users/mhreh/research/MONAI-FL/MONAI-FL/save/models/server/testmodel.pth'
savedir = 'server_model'
checkpointdir = os.path.join('./checkpoints', savedir)
FILE = os.path.join(checkpointdir, 'checkpoint.pth.tar')

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
print("Server is binded")

modelCheckPoint = {
    "epoch": 0,
    "state_dict": {},  
    "optimizer": {},
    "best_metric": 0
    }

GlobalWeights = torch.zeros([])
GlobalEpochs = 3 #args_parser()
Local_Weights = torch.zeros([])

def recvall(n, conn):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
        #time.sleep(0.1)
    return data

def sendMessage(msg, conn):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    conn.send(send_length)
    conn.send(message)

def receiveMessage(conn):
    retmsg = ''
    msg_length = conn.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        retmsg = conn.recv(msg_length).decode(FORMAT)
    return retmsg

def sendModel(conn):
    # the name of file we want to send, make sure it exists
    filename = "checkpoint.pth.tar"
    # get the file size
    filesize = os.path.getsize(FILE)
    # send the filename and filesize
    conn.send(f"{filename}{SEPARATOR}{filesize}".encode())

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
            conn.sendall(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))
            time.sleep(0.0001)
            #print(bytes_read)
    conn.shutdown(socket.SHUT_WR)
    #close the socket
    #conn.close()

def sendWeights(msgData, conn):
    message = pickle.dumps(msgData)
    #print(message)
    msg_length = len(message)
    #print(msg_length)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    conn.send(send_length)
    conn.sendall(message)

def receiveModel(conn):
    received = conn.recv(BUFFER_SIZE).decode()
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
            bytes_read = conn.recv(BUFFER_SIZE)
            time.sleep(0.1)
            if not bytes_read:    
                print("No More Data :-)")
                # file transmitting is done
                break
            # write to the file the bytes we just received
            #print(bytes_read)
            f.write(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))

def receiveWeights(conn):
    msg=''
    msg_length = conn.recv(HEADER).decode(FORMAT)
    print(msg_length)
    if msg_length:
        msg_length = int(msg_length)
        msg = recvall(msg_length, conn)#.decode(FORMAT)
       # print(msg)
       # print(msg_length)
       # print(len(msg))
        #msg = pickle.loads(mssg)
    return msg

def handle_communication(ep_round, conn, addr):
    LocalWeights = torch.zeros([])
    if ep_round == 0:
        print ("This is first round")
        sendMessage("Do you have global model?", conn)
        modelExists = receiveMessage(conn)
        if modelExists == "True":
            # send weights
            print("Sending weights!")
            modelCP = torch.load(FILE)
            #print(modelCP['state_dict'])
            sendWeights(modelCP['state_dict'], conn)
            LocalWeights = receiveWeights(conn)
        else:
            print("Initial Global Model Transferred!")
            sendModel(conn)
            print("Initial Global Model Transferred!")
            LocalWeights = receiveWeights(conn)
    else:
        print ("This is round: ", str(ep_round+1))
        print("Sending weights!")
        modelCP = torch.load(FILE)
        #print(modelCP['state_dict'])
        sendWeights(modelCP['state_dict'], conn)
        LocalWeights = receiveWeights(conn)

            # if ep_round == 0:
        #     #modelCP = torch.load(FILE)
        #     #print(modelCP)
        #     sendMessage(str(ep_round), conn) # 3
        #     sendModel(conn) # 4
        #     print("Model Checkpoint Succeccsfully transferred at Round_ : ", ep_round)
        #     #Local_Weights = receiveWeights(conn)
        # elif ep_round != 0:
        #     print("Im not in first epoch")
        #     # modelCP = torch.load(FILE)
        #     # print(modelCP)
        #     # #print("Server is sending the current global model weights")
        #     # #sendMessage(str(ep_round), conn)
        #     # sendModel(modelCP['model_state'], conn)
        #     # print("Model Weights Succeccsfully transferred")
        #     # Local_Weights = receiveWeights(conn)
        #     # sendMessage(CONNECT_MESSAGE, conn)

    #return Local_Weights
    return LocalWeights

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    glob_epoch = 0

    mssg = receiveMessage(conn) # 1
    print(mssg)

    if mssg == CONNECT_MESSAGE:
        sendMessage("Welcome! You are connected with the sever...", conn) # 2
        print("Starting FL protocol at client with client", str(addr))

        sendMessage(str(GlobalEpochs), conn)

        while glob_epoch < GlobalEpochs:
            print("Global Epoch: "+ str(glob_epoch+1) + "/" + str(GlobalEpochs))
            Local_Weights = handle_communication(glob_epoch, conn, addr)
            #print(Local_Weights)
            GlobalWeights = GlobalWeights.add(Local_Weights)
            glob_epoch += 1
        
        #AvgWeights = FedAvg(GlobalWeights)
    
    sendMessage(DISCONNECT_MESSAGE, conn)
    conn.close()

def start():
    server.listen()
    print(f"[LISTENING] server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()

        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount()-1}")
print("[STARTING] server is starting")

start()

