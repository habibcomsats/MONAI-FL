# This file will store all the communication related functions from the server end. This will be directly communicating with client
import sys
#path for linux distribution
sys.path.insert(1, '/home/habib/myResearch/MONAI-FL')
#path for windows installation
#sys.path.insert(1, 'C:/Users/mhreh/research/MONAI-FL')

import socket
import threading
#import pickle
import torch
import tqdm
import os
import json
import time

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

GLOBAL_MODEL_MESSAGE = "SendGlobalModel"
MODEL_WEIGHTS_MESSAGE = "SendModelWeights"
MODEL_PARAMETERS_MESSAGE = "SendModelParameters"
MODEL_CONFIGURATION_MESSAGE = "SendModelConfigurations"

#path for linux distribution
FILE = '/home/habib/myResearch/MONAI-FL/save/models/server/testmodel.pth'
#path for windows installation
#FILE = 'C:/Users/mhreh/research/MONAI-FL/save/models/server/testmodel.pth'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
print("Server is binded")

modelCheckPoint = {
    "epoch": 0,
    "model_state": {},  
    "optim_state": {}
    }

GlobalWeights = torch.zeros(dtype=float16, requires_grad=True)
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

def sendModelMessage(msg, conn):
    if msg == "global_model":
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
                print(bytes_read)
        conn.shutdown(socket.SHUT_WR)
        #close the socket
        #conn.close()

    elif msg == "global_weights":
        message = json.dumps(msg) #.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        conn.send(send_length)
        conn.send(message)

    elif msg == "global_parameters":
        message = json.dumps(msg) #.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        conn.send(send_length)
        conn.send(message)

    elif msg == "global_configurations":
        message = json.dumps(msg) #.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        conn.send(send_length)
        conn.send(message)

def receiveModelMessage(msg, conn):
    if msg == "local_model":
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
                print(bytes_read)
                progress.update(len(bytes_read))

    elif msg == "local_weights":
        message = json.dumps(msg) #.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        conn.send(send_length)
        conn.send(message)

    elif msg == "local_parameters":
        message = json.dumps(msg) #.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        conn.send(send_length)
        conn.send(message)

    elif msg == "local_configurations":
        message = json.dumps(msg) #.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        conn.send(send_length)
        conn.send(message)

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    while connected:
        mssg = receiveMessage(conn)
        if mssg == CONNECT_MESSAGE:
            sendMessage("Welcome! You are connected with the sever...", conn)
            message = receiveMessage(conn)
            print(message)
            while message != DISCONNECT_MESSAGE:
                modelCP = torch.load(FILE)
                if message == GLOBAL_MODEL_MESSAGE:
                    print("Server is sending the current global model")
                    sendModelMessage(modelCP, conn)
                    print("Model Succeccsfully transferred")
                    Local_Weights = receiveModelMessage(GLOBAL_WEIGHTS_MESSAGE, conn)
                    GlobalWeights = GlobalWeights.add(Local_Weights)
                elif message == MODEL_WEIGHTS_MESSAGE:
                    print("Server is sending the current global model checkpoint")
                    sendModelMessage(modelCP['model_state'], conn)
                    print("Model Succeccsfully transferred")
                elif message == MODEL_PARAMETERS_MESSAGE:
                    print("Server is sending the current global model checkpoint")
                    sendModelMessage(modelCP['optim_state'], conn)
                    print("Model Succeccsfully transferred")
                elif message == MODEL_CONFIGURATION_MESSAGE:
                    print("Server is sending the current global model checkpoint")
                    sendModelMessage(modelCP['epoch'], conn)
                    print("Model Succeccsfully transferred")
                message = receiveMessage(conn)
        connected = False  
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

