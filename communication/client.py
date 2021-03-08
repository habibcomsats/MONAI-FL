import socket
import tqdm

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

def receiveModel():
    received = client_socket.recv(BUFFER_SIZE).decode()
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
            bytes_read = client_socket.recv(BUFFER_SIZE)
            time.sleep(0.000001)
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
        




send("Connected")

receiveModel()

send("Hello Everyone")
input()

send("Hello Habib")

send(DISCONNECT_MESSAGE)