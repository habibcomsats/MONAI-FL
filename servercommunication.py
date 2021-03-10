# This file will store all the communication related functions from the server end. This will be directly communicating with client
import sys
#path for linux distribution
# sys.path.insert(1, '/home/habib/myResearch/MONAI-FL')
#path for windows installation
sys.path.insert(1, 'C:/Users/mhreh/research/MONAI-FL')

import socket
import threading
#import pickle
import torch
import tqdm
import os
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

GLOBAL_MODEL_MESSAGE = "SendGlobalModel"
MODEL_CHECKPOINT_MESSAGE = "SendModelCheckPoint"
FILE = 'C:/Users/mhreh/research/MONAI-FL/save/models/server/testmodel.pth'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
print("Server is binded")

def sendMessage(msg, conn):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    conn.send(send_length)
    conn.send(message)

def sendDictMessage(msg, conn):
    message = json.dumps(msg) #.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    conn.send(send_length)
    conn.send(message)

def sendTrainedModel(conn):
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
def receiveMessage(conn):
    retmsg = ''
    msg_length = conn.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        retmsg = conn.recv(msg_length).decode(FORMAT)
    return retmsg
  
def receiveModelCheckpoints(conn, addr):
    connected = True
    while connected:
        msg_length = client.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = client.recv(msg_length).decode(FORMAT)
            if msg == CONNECT_MESSAGE:
                print("Welcome, you are connected to the server")
                print("Server is sending the current model")
                #modelBoostrap()
                #sendModel(conn)
            elif msg == DISCONNECT_MESSAGE:
                connected = False
            
            print(f"[{addr}] {msg}")
            #conn.send("Msg received".encode(FORMAT))

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    while connected:
        mssg = receiveMessage(conn)
        modelCheckPoint = torch.load(FILE)
        if mssg == CONNECT_MESSAGE:
            sendMessage(mssg, conn)
            message = receiveMessage(conn)
            print(message)
            if message == GLOBAL_MODEL_MESSAGE:
                print("Server is sending the current global model")
                sendTrainedModel(conn)
                sendTrainingConfigurations(conn)
            elif message == MODEL_CHECKPOINT_MESSAGE:
                print("Server is sending the current global model checkpoint")
                sendTrainedModel(conn)
                print("Model Succeccsfully transferred")
        elif mssg == DISCONNECT_MESSAGE:
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

