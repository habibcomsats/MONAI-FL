from configparser import ConfigParser

def setNetworkConfigurations():
    
    serverip = '127.0.0.1'
    serverport = '8000'

    clientip1 = '127.0.0.1'
    clientport1 = '8001'

    clientip2 = '127.0.0.1'
    clientport2 = '8001'

    clientip3 = '127.0.0.1'
    clientport3 = '8001'

    parser = ConfigParser()

    parser.add_section('localserver')
    parser.set('localserver', 'ip', serverip)
    parser.set('localserver', 'port', serverport)
    
    parser.add_section('localclient')
    parser.set('localclient', 'ip1', clientip1)
    parser.set('localclient', 'port1', clientport1)

    parser.set('localclient', 'ip2', clientip2)
    parser.set('localclient', 'port2', clientport2)

    parser.set('localclient', 'ip3', clientip3)
    parser.set('localclient', 'port3', clientport3)

    fp=open('netconfig.ini','w')
    parser.write(fp)
    fp.close()

setNetworkConfigurations()

def getNetworkConfigurations():
    fl_participants = list()

    parser = ConfigParser()
    parser.read('netconfig.ini')

    serverip = parser.get('localserver','ip')
    serverport = parser.get('localserver','port')
    server = serverip, serverport
    fl_participants.append(server)

    clientip1 = parser.get('localclient', 'ip1')
    clientport1 = parser.get('localclient', 'port1')
    client1 = clientip1, clientport1
    fl_participants.append(client1)

    clientip2 = parser.get('localclient', 'ip2')
    clientport2 = parser.get('localclient', 'port2')
    client2 = clientip2, clientport2
    fl_participants.append(client2)

    clientip3 = parser.get('localclient', 'ip3')
    clientport3 = parser.get('localclient', 'port3')
    client3 = clientip3, clientport3
    fl_participants.append(client3)

    return fl_participants
