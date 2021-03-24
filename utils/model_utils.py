import tensorflow.keras.layers as layers


def add_layer(x, ch, drop, architecture, arch_params):

    # TODO Add lots of comments and docstrings
    if architecture == 'conv_1D':
        conv_size = arch_params.conv_size
        x = conv_bn_activate(x, ch, conv_size=conv_size, drop=drop)
    elif architecture == 'attention':
        x = attention_bn_activate(x, ch, query_dim=ch, drop=drop)
    elif architecture == 'lstm':
        x = dense_bn_activate(x, ch, drop=drop)  # LSTM also uses fc except first layer
    elif architecture == 'fully_connected':
        x = dense_bn_activate(x, ch, drop=drop)
    else:
        raise NotImplementedError
    return x


def dense_bn_activate(x, out_dim, activation='relu', drop=0.):
    x = layers.Dense(out_dim)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if drop > 0:
        x = layers.Dropout(rate=drop)(x)
    return x


def conv_bn_activate(x, out_dim, activation='relu', conv_size=3, drop=0.):
    x = layers.Conv1D(out_dim, conv_size)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, 2)(x)
    if drop > 0:
        x = layers.Dropout(rate=drop)(x)
    return x


def attention(x, query_dim, out_dim):
    q = layers.Conv1D(query_dim, 1)(x)
    v = layers.Conv1D(query_dim, 1)(x)
    attn = layers.Attention()([q, v])
    outs = layers.Conv1D(out_dim, 1)(attn)
    return outs


def attention_bn_activate(x, out_dim, query_dim, activation='relu', drop=0.):
    x = attention(x, query_dim, out_dim)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = dense_bn_activate(x, out_dim, drop=drop)(x)
    x = layers.MaxPooling1D(2, 2)(x)
    if drop > 0:
        x = layers.Dropout(rate=drop)(x)
    return x


def freeze_model_except_last_layer(model):
    for idx, layer in enumerate(model.layers[:-1]):
        if not isinstance(layer, layers.BatchNormalization):
            model.layers[idx].trainable = False


def unfreeze_model_except_last_layer(model):
    for idx, layer in enumerate(model.layers[:-1]):
        if not isinstance(layer, layers.BatchNormalization):
            model.layers[idx].trainable = True
