import socket
import threading
import pickle
import torch
import tqdm
import os

from communication.network import getNetworkConfigurations
#from utils.options import args_parser
#from flapi.monai_fl_main import modelBoostrap

#args = args_parser()
#args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

#dataset_train = []
#dataset_test = []
#img_size = []

#modelCheckPoint = {
#"epoch": 0,
#    "model_state": {},  
#    "optim_state": {}
#    }

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

FILE = 'C:/Users/mhreh/research/MONAI-FL/save/models/server/testmodel.pth'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
print("Server is binded")

def sendModel(conn):
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
    conn.shutdown(socket.SHUT_WR)
    # close the socket
    #conn.close()

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

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
                sendModel(conn)
            elif msg == DISCONNECT_MESSAGE:
                connected = False
            
            print(f"[{addr}] {msg}")
            #conn.send("Msg received".encode(FORMAT))
    
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
