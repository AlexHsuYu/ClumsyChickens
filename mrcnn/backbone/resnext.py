import keras.backend as KB
import keras.layers as KL

############################################################
#  ResNeXt Graph
############################################################

def identity_block(input_tensor, kernel_size, filters, stage, block, cardinality=1, use_bias=True, train_bn=True):
    """
        Conv block without convolution for shortcut.
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # group_list = []
    result = input_tensor
    for c in range(cardinality):
        x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a_' + str(c + 1), use_bias=use_bias)(input_tensor)
        x = KL.BatchNormalization(name=bn_name_base + '2a_' + str(c + 1))(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b_' + str(c + 1),
                      use_bias=use_bias)(x)
        x = KL.BatchNormalization(name=bn_name_base + '2b_' + str(c + 1))(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c_' + str(c + 1), use_bias=use_bias)(x)
        x = KL.BatchNormalization(name=bn_name_base + '2c_' + str(c + 1))(x, training=train_bn)
        # group_list.append(x)

        if (c == 0):
            result = x
        else:
            result = KL.Add()([x, result])

    # x = KL.Add()(group_list)
    x = result
    # x = result

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), cardinality=1, use_bias=True, train_bn=True):
    """
        Equivalent group conv block. Implemented so because Keras does not support group conv.
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # group_list = []
    result = input_tensor
    for c in range(cardinality):
        x = KL.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a_' + str(c+1), use_bias=use_bias)(input_tensor)
        x = KL.BatchNormalization(name=bn_name_base + '2a_' + str(c+1))(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b_' + str(c+1), use_bias=use_bias)(x)
        x = KL.BatchNormalization(name=bn_name_base + '2b_' + str(c+1))(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c_' + str(c+1), use_bias=use_bias)(x)
        x = KL.BatchNormalization(name=bn_name_base + '2c_' + str(c+1))(x, training=train_bn)
        # group_list.append(x)

        if (c == 0):
            result = x
        else:
            result = KL.Add()([x, result])

    # x = KL.Add()(group_list)
    x = result

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1_' + str(c+1), use_bias=use_bias)(input_tensor)
    shortcut = KL.BatchNormalization(name=bn_name_base + '1_' + str(c+1))(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnext_graph(input_image, architecture, stage5=False, train_bn=True):
    """
        Build the ResNeXt graph
    """
    # Currently only works for cardinality=1 (equivalent to ResNet)! Other values give shape mismatch!

    assert architecture in ["resnext50", "resnext101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = KL.BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [4, 4, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [4, 4, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [4, 4, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [256, 256, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [256, 256, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4
    x = conv_block(x, 3, [512, 512, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnext50": 5, "resnext101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [512, 512, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x

    # Stage 5
    if stage5:
        x = conv_block(x, 3, [1024, 1024, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [1024, 1024, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [1024, 1024, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None

    return [C1, C2, C3, C4, C5]
