import sys
#path for linux distribution
# sys.path.insert(1, '/home/habib/myResearch/MONAI-FL')
#path for windows installation
sys.path.insert(1, 'C:/Users/mhreh/research/MONAI-FL')

import socket
import os
import tqdm
import time
import json

from flapi.client_fl import modelBoostrap

HEADER  = 64
PORT = 8000
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
CONNECT_MESSAGE = "Connected"

GLOBAL_MODEL_MESSAGE = "SendGlobalModel"
MODEL_CHECKPOINT_MESSAGE = "SendModelCheckPoint"
# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

SERVER = "127.0.0.1"  
ADDR = (SERVER,PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

FILE = 'C:/Users/mhreh/research/MONAI-FL/save/models/client/testmodel.pth'

def sendMessage(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    client.send(send_length)
    client.send(message)

def sendModelWeights(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    client.send(send_length)
    client.send(message)
    #print(client.recv(2048).decode(FORMAT))

def sendModelParameters(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    client.send(send_length)
    client.send(message)
    #print(client.recv(2048).decode(FORMAT))

def sendModelTrainingConfigurations(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    client.send(send_length)
    client.send(message)
    #print(client.recv(2048).decode(FORMAT))

def receiveMessage():
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode(FORMAT)
    return msg

def receiveDictMessage():
    msg_length = client.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length)#.decode(FORMAT)
        msg = json.loads(msg)
    return msg


def receiveModelArchitecture():
    return

def receiveGlobalModel():
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
                print("Nothing is received")
                # file transmitting is done
                break
            # write to the file the bytes we just received
            #print(bytes_read)
            f.write(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))

    # close the client socket
    #client_socket.close()
    # close the server socket
    #client.close()
    # 

def receiveCheckpointMessage():
    return 
   
def receiveGlobalModelTrainingConfigurations():
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
   
def handle_server():
    print("Connected with Server...")
    message  = sendMessage(CONNECT_MESSAGE)
    message  = receiveMessage()
    print("Server is ", message)

    model = modelBoostrap 
    if model:
        print("i have model")
        sendMessage(MODEL_CHECKPOINT_MESSAGE)
        Checkpoint = receiveCheckpointMessage()
        print(weights)
        sendMessage(MODEL_PARAMETER_MESSAGE)
        modelParameters = receiveDictMessage()
        print(modelParameters)
        sendMessage(MODEL_CONFIGURATIONS_MESSAGE)
        epochs = receiveMessage
        print(epochs)
    else:
        sendMessage(GLOBAL_MODEL_MESSAGE)
        

    # connected = True
    # while connected:
    #     if message == CONNECT_MESSAGE:
    #         message = receiveMessage()
    #         print(message)
    #         # model = modelBoostrap  
    
    #         # msg_length = client.recv(HEADER).decode(FORMAT)
    #         # print(msg_length)
    #         # if msg_length:
    #         #     msg_length = int(msg_length)
    #         #     msg = client.recv(msg_length).decode(FORMAT)
            
    #         #         #modelBoostrap()
    #         #         #sendTrainedModel(conn)
    #     elif msg == DISCONNECT_MESSAGE:
    #         connected = False

handle_server()

# sendMessage(CONNECT_MESSAGE)
# sendMessage(MODEL_ARCHITECTURE_MESSAGE)
# sendMessage(GLOBAL_MODEL_MESSAGE)
# sendMessage(MODEL_WEIGHTS_MESSAGE)
# sendMessage(MODEL_PARAMETER_MESSAGE)
# sendMessage(MODEL_CONFIGURATIONS_MESSAGE)

# #receiveGlobalModel()

# sendMessage("Hello Everyone")
# input()

# sendMessage("Hello Habib")

# sendMessage(DISCONNECT_MESSAGE)