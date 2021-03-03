from configparser import ConfigParser

def selectDevices():
    parser = ConfigParser()
    parser.read('netconfig.ini')

    serverip = parser['localserver']['ip']
    serverport = parser['localserver']['port']
    server = (serverip, serverport)
    print("Local Server address is, ", server)
   
    clientip = parser.get('localclient', 'ip')
    clientport = parser.get('localclient', 'port')
    client = (clientip, clientport)
    print("Local Client address is, ", client)
    
    collectAvailableDevices = True
    # collectDevicesList = CollectFromConfigurationFile
    # startListeningOnServer = False
    # collectAvailableDevices = checkDeviceStatus # by sending and receiving acknowledgement messages to the devices
    return collectAvailableDevices