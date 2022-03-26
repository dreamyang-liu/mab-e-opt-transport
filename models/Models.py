from tensorflow.keras import layers, Model, Sequential


class Conv1d_Model(Model):

    def __init__(self, input_dim, class_dim, arch_params):
        super().__init__()
        self.model = Sequential()
        self.model.add(layers.Input(input_dim))
        self.model.add(layers.BatchNormalization())
        self.make_layers(arch_params)
        self.model.add(layers.Flatten())
        self.head = layers.Dense(class_dim, activation='softmax')

    
    def make_layers(self, arch_params):
        conv_size = arch_params.conv_size
        conv_channels = arch_params.conv_channels
        drop_ratio = arch_params.drop_ratio
        for  ch in conv_channels:
            self.model.add(layers.Conv1D(ch, conv_size))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.MaxPool1D(2, 2))
            self.model.add(layers.Activation('relu'))
            if drop_ratio > 0:
                self.model.add(layers.Dropout(drop_ratio))
    
    def call(self, x):
        o = self.model(x)
        o = self.head(o)
        return o
    
    def compute_features(self, x):
        feat = self.model(x)
        return feat
