import torch

import torch.nn as nn

# can be used to save any dictionary object, it uses python's pickle for serialization

torch.save(args, PATH)

torch.load(PATH)

model.load_state_dict(args)


# lazzzy method
# the downside is, it saves paths and directory structure which may create issues dueing model portability.
torch.save(model, PATH)
model = torch.load(PATH)
model.eval()

# recommend method
##state dict

torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()



