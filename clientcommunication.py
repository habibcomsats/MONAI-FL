# This file will store all the communication related functions from the client end. This will be directly communicating with server

import sys
#path for linux distribution
sys.path.insert(1, '/home/habib/myResearch/MONAI-FL')
#path for windows installation
#sys.path.insert(1, 'C:/Users/mhreh/research/MONAI-FL')

import socket
import os
import tqdm
import time
import json

from clientfilehandler import modelBootstrap

HEADER  = 64
PORT = 8000
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
CONNECT_MESSAGE = "Connected"

GLOBAL_MODEL_MESSAGE = "SendGlobalModel"
MODEL_WEIGHTS_MESSAGE = "SendModelWeights"
MODEL_PARAMETERS_MESSAGE = "SendModelParameters"
MODEL_CONFIGURATION_MESSAGE = "SendModelConfigurations"

# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

SERVER = "127.0.0.1"  
ADDR = (SERVER,PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

#path for linux distribution
FILE = '/home/habib/myResearch/MONAI-FL/save/models/server/testmodel.pth'
#path for windows installation
#FILE = 'C:/Users/mhreh/research/MONAI-FL/save/models/server/testmodel.pth'

modelCheckPoint = {
    "epoch": 0,
    "model_state": {},  
    "optim_state": {}
    }

def sendMessage(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    client.send(send_length)
    client.send(message)

def receiveMessage():
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
    return msg

def sendModelMessage(msg):
    if msg == "local_model":
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
                print(bytes_read)
        client.shutdown(socket.SHUT_WR)
        #close the socket
        #conn.close()
    elif msg == "local_weights":
        message = json.dumps(msg) #.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        client.send(send_length)
        client.send(message)
    elif msg == "local_parameters":
        message = json.dumps(msg) #.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        client.send(send_length)
        client.send(message)
    elif msg == "local_configurations":
        message = json.dumps(msg) #.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*(HEADER-len(send_length))
        client.send(send_length)
        client.send(message)
    return 

def receiveModelMessage(msg):
    if msg == "global_model":
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

    elif msg == "global_weights":
        msg_length = client.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = client.recv(msg_length)#.decode(FORMAT)
            msg = json.loads(msg)
        return msg

    elif msg == "global_parameters":
        msg_length = client.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = client.recv(msg_length)#.decode(FORMAT)
            msg = json.loads(msg)
        return msg

    elif msg == "global_configurations":
        msg_length = client.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = client.recv(msg_length)#.decode(FORMAT)
            msg = json.loads(msg)
        return msg
    
def handle_server():
    sendMessage(CONNECT_MESSAGE)
    server_message  = receiveMessage()
    print(server_message)

    model = modelBootstrap 
    if model:
        print("i have model")
        sendMessage(GLOBAL_WEIGHTS_MESSAGE)
        Local_Weights = receiveModelMessage(GLOBAL_WEIGHTS_MESSAGE)
    else:
        sendMessage(GLOBAL_MODEL_MESSAGE)
        
handle_server()


