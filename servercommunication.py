# This file will store all the communication related functions from the server end. This will be directly communicating with client
import sys
#path for linux distribution
#sys.path.insert(1, '/home/habib/myResearch/MONAI-FL')

#path for windows installation
sys.path.insert(1, 'C:/Users/mhreh/research/MONAI-FL/MONAI-FL/')

import socket
import threading
import pickle
import torch
import tqdm
import os
import time
import json

from serverfilehandler import modelBootstrap
from communication.network import getNetworkConfigurations
from utils.options import args_parser

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
FILE = 'C:/Users/mhreh/research/MONAI-FL/MONAI-FL/save/models/server/testmodel.pth'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
print("Server is binded")

modelCheckPoint = {
    "epoch": 0,
    "model_state": {},  
    "optim_state": {}
    }

GlobalWeights = torch.tensor([])
GlobalEpochs = 3 #args_parser()
Local_Weights = torch.tensor([])

def recvall(n, conn):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
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

def sendModelMessage(msg, msgData, conn):
    if msg == "model":
        # the name of file we want to send, make sure it exists
        filename = "testmodel.pth"
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
                #print(bytes_read)
        conn.shutdown(socket.SHUT_WR)
        #close the socket
        #conn.close()

    elif msg == "weights":
        message = pickle.dumps(msgData)
        print(message)
        msg_length = len(message)
        print(msg_length)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        conn.send(send_length)
        conn.sendall(message)

def receiveModelMessage(msg, conn):
    if msg == "model":
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
                time.sleep(0.00000001)
                if not bytes_read:    
                    print("No More Data :-)")
                    # file transmitting is done
                    break
                # write to the file the bytes we just received
                #print(bytes_read)
                f.write(bytes_read)
                # update the progress bar
                progress.update(len(bytes_read))

    elif msg == "weights":
        msg_length = conn.recv(HEADER).decode(FORMAT)
        print(msg_length)
        if msg_length:
            msg_length = int(msg_length)
            mssg = recvall(msg_length, conn)#.decode(FORMAT)
            print(mssg)
            print(msg_length)
            print(len(mssg))
            #msg = pickle.loads(mssg)
        return msg

def handle_communication(ep_round, conn, addr):
    mssg = receiveMessage(conn)
    print(mssg)
    if mssg == CONNECT_MESSAGE:
        sendMessage("Welcome! You are connected with the sever...", conn)
        print("Starting FL protocol at client with client", str(addr))
        if ep_round == 0:
            modelCP = torch.load(FILE)
            print(modelCP)
            sendMessage(str(ep_round), conn)
            sendModelMessage(MODEL_MESSAGE, modelCP, conn)
            print("Model Checkpoint Succeccsfully transferred at Round_ : ", ep_round)
            Local_Weights = receiveModelMessage(WEIGHTS_MESSAGE, conn)
        else:
            modelCP = torch.load(FILE)
            print(modelCP)
            #print("Server is sending the current global model weights")
            sendMessage(str(ep_round), conn)
            sendModelMessage(WEIGHTS_MESSAGE, modelCP['model_state'], conn)
            print("Model Weights Succeccsfully transferred")
            Local_Weights = receiveModelMessage(WEIGHTS_MESSAGE, conn)

    return Local_Weights

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    glob_epoch = 0
    while glob_epoch < GlobalEpochs:
        Local_Weights = handle_communication(glob_epoch, conn, addr)
        # GlobalWeights = GlobalWeights.add(Local_Weights)
        print(glob_epoch)
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

