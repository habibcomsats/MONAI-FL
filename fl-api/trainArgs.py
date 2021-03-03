from configparser import ConfigParser

parser = ConfigParser()
parser.read('trainconfig.ini')

def selectFederatedArguments():
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
    return True

def selectModelArguments():
    # [ModelArguments]
    model =             = parser['ModelArguments']['model']
    number_of_kernels   = parser['ModelArguments']['number_of_kernels']
    kernel_sizes        = parser['ModelArguments']['kernel_sizes']
    norms               = parser['ModelArguments']['norms']
    number_of_filters   = parser['ModelArguments']['number_of_filters']
    max_pool            = parser['ModelArguments']['max_pool']
    return True

def selectOtherArguments():
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
    return True