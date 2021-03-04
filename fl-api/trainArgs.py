from configparser import ConfigParser

parser = ConfigParser()
parser.read('trainconfig.ini')

def selectFederatedArguments():
    
    parser = ConfigParser()
    parser.read('trainconfig.ini')

    # [FederatedArguments]
    epochs              = parser['FederatedArguments']['epochs']
    workers             = parser['FederatedArguments']['workers']
    fraction            = parser['FederatedArguments']['fraction']
    local_epochs        = parser['FederatedArguments']['local_epochs']
    local_batch_size    = parser['FederatedArguments']['local_batch_size']
    test_batch_size     = parser['FederatedArguments']['test_batch_size']
    learning_rate       = parser['FederatedArguments']['learning_rate']
    momentum            = parser['FederatedArguments']['momentum']
    split               = parser['FederatedArguments']['split']
    FedArgs = (epochs, workers, fraction, local_epochs, local_batch_size, test_batch_size, learning_rate, momentum, split)
    print(FedArgs)
    return True

selectFederatedArguments()

def selectModelArguments():
    
    parser = ConfigParser()
    parser.read('trainconfig.ini')
    
    # [ModelArguments]
    model               = parser['ModelArguments']['model']
    number_of_kernels   = parser['ModelArguments']['number_of_kernels']
    kernel_sizes        = parser['ModelArguments']['kernel_sizes']
    norms               = parser['ModelArguments']['norms']
    number_of_filters   = parser['ModelArguments']['number_of_filters']
    max_pool            = parser['ModelArguments']['max_pool']
    ModelArgs = (model, number_of_kernels, kernel_sizes, norms, number_of_filters, max_pool)
    print(ModelArgs)
    return True

selectModelArguments()

def selectOtherArguments():

    parser = ConfigParser()
    parser.read('trainconfig.ini')

    # [OtherArguments]
    dataset             = parser['OtherArguments']['dataset']
    iid                 = parser['OtherArguments']['iid']
    number_of_classes   = parser['OtherArguments']['number_of_classes']
    number_of_channels  = parser['OtherArguments']['number_of_channels']
    gpu                 = parser['OtherArguments']['gpu']
    stopping_rounds     = parser['OtherArguments']['stopping_rounds']
    verbose             = parser['OtherArguments']['verbose']
    seed                = parser['OtherArguments']['seed']
    all_clients         = parser['OtherArguments']['all_clients']
    OthArgs = (dataset, iid, number_of_classes, number_of_channels, gpu, stopping_rounds, verbose, seed, all_clients)
    print(OthArgs)
    return True

selectModelArguments()