import socket

HEADER  = 64
PORT = 8000
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
CONNECT_MESSAGE = "Connected"

SERVER = "127.0.0.1"  
ADDR = (SERVER,PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

FILE = 'C:/Users/mhreh/research/MONAI-FL/save/models/client/testmodel.pth'

def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' '*(HEADER-len(send_length))
    client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))

def receiveModel(filepath):
  #File = "model.pth"
    mcp = client.recv(2048).decode(FORMAT)
    print(mcp)
    # modelCheckPoint = {
    #     "epoch": 90,
    #     "model_state": model.state_dict(),  
    #     "optim_state": optimizer.state_dict()
    #     }
    # print(modelCheckPoint)
    # torch.save(modelCheckPoint, FILE)
        
def handle_server():
    #print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    while connected:
        msg_length = client.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = client.recv(msg_length)#.decode(FORMAT)
            #if msg == CONNECT_MESSAGE:
            #   print("Client is waiting for model")
            #    print("client is receving model")
            receiveModel(FILE)
            #elif msg == DISCONNECT_MESSAGE:
            connected = False
            
            #print(f"[{addr}] {msg}")
            client.send("Msg received".encode(FORMAT))
    
   #conn.close()




send("Connected")

handle_server()

send("Hello Everyone")
input()

send("Hello Habib")

send(DISCONNECT_MESSAGE)