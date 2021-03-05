from configparser import ConfigParser

def setFederatedArguments():
    #Federeated Arguments
    epochs              = '10'      # number of training rounds
    workers             = '2'       # number of workkers for federated rounds
    fraction            = '0.1'     # the fraction of clients
    local_epochs        = '5'       # number of local epochs
    local_batch_size    = '10'      # local batch size
    test_batch_size     = '128'     # test batch size
    learning_rate       = '0.01'    # learning rate
    momentum            = '0.5'     # SDG momentum (default: 0.5)
    split               = 'user'    # train-test split type, user or sample

    parser = ConfigParser()

    parser.add_section('FederatedArguments')
    parser.set('FederatedArguments', 'ip', epochs)
    parser.set('FederatedArguments', 'workers', workers)
    parser.set('FederatedArguments', 'fraction', fraction)
    parser.set('FederatedArguments', 'local_epochs', local_epochs)
    parser.set('FederatedArguments', 'local_batch_size', local_batch_size)
    parser.set('FederatedArguments', 'test_batch_size', test_batch_size)
    parser.set('FederatedArguments', 'learning_rate', learning_rate)
    parser.set('FederatedArguments', 'momentum', momentum)
    parser.set('FederatedArguments', 'split', split)

    fp=open('netconfig.ini','w')
    parser.write(fp)
    fp.close()

def getFederatedArguments():
    federatedArguments = list()
    parser = ConfigParser()
    parser.read('trainconfig.ini')
    epochs              = parser.get('FederatedArguments','epochs')
    workers             = parser.get('FederatedArguments','workers')
    fraction            = parser.get('FederatedArguments','fraction')
    local_epochs        = parser.get('FederatedArguments','local_epochs')
    local_batch_size    = parser.get('FederatedArguments','local_batch_size')
    test_batch_size     = parser.get('FederatedArguments','test_batch_size')
    learning_rate       = parser.get('FederatedArguments','learning_rate')
    momentum            = parser.get('FederatedArguments','momentum')
    split               = parser.get('FederatedArguments','split')
    federatedArguments = (epochs, workers, fraction, local_epochs, local_batch_size, test_batch_size, learning_rate, momentum, split)
    
    return federatedArguments

def setModelArguments():

    # [ModelArguments]
    model = 'mlp'           # model name
    number_of_kernels = '9'   # number of each kind of kernel
    kernel_sizes = '3,4,5'  # comma-separated kernel size to use for convolution
    norms = 'batch_norm'      # batch_norm, layer_norm, or None
    number_of_filters = '32'  # number of filters for convolutional networks
    max_pool = 'True'         # Whether use max pooling rather than strided convolutions

    parser = ConfigParser()

    parser.add_section('ModelArguments')
    parser.set('ModelArguments', 'model', model)
    parser.set('ModelArguments', 'number_of_kernels', number_of_kernels)
    parser.set('ModelArguments', 'kernel_sizes', kernel_sizes)
    parser.set('ModelArguments', 'norms', norms)
    parser.set('ModelArguments', 'number_of_filters', number_of_filters)
    parser.set('ModelArguments', 'max_pool', max_pool)

    fp=open('netconfig.ini','w')
    parser.write(fp)
    fp.close()

def getModelArguments():
    modelArguments = list()
    parser = ConfigParser()
    parser.read('trainconfig.ini')
    model               = parser.get('ModelArguments', 'model')
    number_of_kernels   = parser.get('ModelArguments', 'number_of_kernels')
    kernel_sizes        = parser.get('ModelArguments', 'kernel_sizes')
    norms               = parser.get('ModelArguments', 'norms')
    number_of_filters   = parser.get('ModelArguments', 'number_of_filters')
    max_pool            = parser.get('ModelArguments', 'max_pool')
    modelArguments      = (model, number_of_kernels, kernel_sizes, norms, number_of_filters, max_pool)
    
    return modelArguments

def setOtherArguments():

    # [OtherArguments]
    dataset =               'mnist'   # name of dataset
    iid =                   'True'    # whether the dataset needs to be iid or non_iid
    number_of_classes =     '10'      # number of classes
    number_of_channels =    '3'       # number of channels of images
    gpu =                   '0'       # GPU ID, -1 for CPU
    stopping_rounds =       '10'     # rounds of early stopping 
    verbose =               'True'    # verbose print
    seed =                  '1'       # random seed (default: 1)
    all_clients =           'True'   # aggregation over all clients

    parser = ConfigParser()

    parser.add_section('OtherArguments')
    parser.set('OtherArguments', 'dataset', dataset)
    parser.set('OtherArguments', 'iid', iid)
    parser.set('OtherArguments', 'number_of_classes', number_of_classes)
    parser.set('OtherArguments', 'number_of_channels', number_of_channels)
    parser.set('OtherArguments', 'gpu', gpu)
    parser.set('OtherArguments', 'stopping_rounds', stopping_rounds)
    parser.set('OtherArguments', 'verbose', verbose)
    parser.set('OtherArguments', 'seed', seed)
    parser.set('OtherArguments', 'all_clients', all_clients)

    fp=open('netconfig.ini','w')
    parser.write(fp)
    fp.close()


def getOtherArguments():

    parser = ConfigParser()
    parser.read('trainconfig.ini')

    # [OtherArguments]
    dataset             = parser.get('OtherArguments','dataset')
    iid                 = parser.get('OtherArguments','iid')
    number_of_classes   = parser.get('OtherArguments','number_of_classes')
    number_of_channels  = parser.get('OtherArguments','number_of_channels')
    gpu                 = parser.get('OtherArguments','gpu')
    stopping_rounds     = parser.get('OtherArguments','stopping_rounds')
    verbose             = parser.get('OtherArguments','verbose')
    seed                = parser.get('OtherArguments','seed')
    all_clients         = parser.get('OtherArguments','all_clients')
    OthArgs = (dataset, iid, number_of_classes, number_of_channels, gpu, stopping_rounds, verbose, seed, all_clients)
    return OthArgs
setFederatedArguments()
setModelArguments()
setOtherArguments()
print(getFederatedArguments())
print(getModelArguments())
print(getOtherArguments())
