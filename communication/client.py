import socket
import os
import tqdm
import time

HEADER  = 64
PORT = 8000
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
CONNECT_MESSAGE = "Connected"
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
    #print(client.recv(2048).decode(FORMAT))

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
    
def receiveModelArchitecture():
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
def receiveGlobalModelWeights():
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
   

def receiveGlobalModelParameters():
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
    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == CONNECT_MESSAGE:
                print("Welcome, you are connected to the server")
                print("Server is sending the current model")
                #modelBoostrap()
                sendTrainedModel(conn)
            elif msg == DISCONNECT_MESSAGE:
                connected = False
            
            print(f"[{addr}] {msg}")
            #conn.send("Msg received".encode(FORMAT))
    
    conn.close()
send("Connected")

receiveGlobalModel()

send("Hello Everyone")
input()

send("Hello Habib")

send(DISCONNECT_MESSAGE)