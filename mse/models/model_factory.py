from conv1d import *
from full_connected import *

# conv1d model
def get_model_conv1d_0():
    model = Conv1d(
        input_dim=3, 
        class_dim=4, 
        hidden_dim=256, 
        dropout=1, 
        kernel_size=2, 
        layers=3
    )
    return model

# full connected model
def get_full_connected_0():
    model = FullConnected(
        input_dim=14 * 2,
        class_dim=4,
        layers=[256, 512, 128]
    )
    return model
