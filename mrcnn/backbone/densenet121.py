import keras.layers as KL
import keras.backend as K
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation


############################################################
#  DenseNet Graph
############################################################

def conv_block(x, stage, branch, number_of_filters, dropout_rate=None, weight_decay=1e-4):
    """
        BN - ReLU - 1x1 Conv - BN - ReLU - 3x3 Conv
    """
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = number_of_filters * 4
    x = KL.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = KL.Activation('relu', name=relu_name_base+'_x1')(x)
    x = KL.Conv2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = KL.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = KL.Activation('relu', name=relu_name_base+'_x2')(x)
    x = KL.ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = KL.Conv2D(number_of_filters, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, number_of_filters, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    """
        BN - Conv (with compression) - AvgPool
    """

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = KL.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = KL.Activation('relu', name=relu_name_base)(x)
    x = KL.Conv2D(int(number_of_filters * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = KL.AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, number_of_layers, number_of_filters, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_number_of_filterss=True):
    """
        Add dense connections by feed output of conv block into subsequent ones.
    """

    concat_feat = x
    for i in range(number_of_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_number_of_filterss:
            number_of_filters += growth_rate

    return concat_feat, number_of_filters


def densenet121_graph(input_tensor, number_of_dense_blocks=4, growth_rate=32, number_of_filters=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=2, weights_path=None):
    """
        Build the DenseNet121 graph
    """

    eps = 1.1e-5
    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
    else:
      concat_axis = 1

    number_of_layers = [6, 12, 24, 16] # For DenseNet-121
    bottom_up_layers = [0, 0, 0, 0, 0]

    # Initial convolution
    x = KL.ZeroPadding2D((3, 3), name='conv1_zeropadding')(input_tensor)
    x = KL.Conv2D(number_of_filters, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = KL.BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    bottom_up_layers[0] = x = Activation('relu', name='relu1')(x)
    x = KL.ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(number_of_dense_blocks - 1):
        stage = block_idx+2
        x, number_of_filters = dense_block(x, stage, number_of_layers[block_idx], number_of_filters, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        bottom_up_layers[stage - 1] = x

        # Add transition_block
        x = transition_block(x, stage, number_of_filters, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        number_of_filters = int(number_of_filters * compression)

    final_stage = stage + 1
    x, number_of_filters = dense_block(x, final_stage, number_of_layers[-1], number_of_filters, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = KL.BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    bottom_up_layers[final_stage-1] = x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = KL.GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    x = Dense(classes, name='fc6')(x)
    x = KL.Activation('softmax', name='prob')(x)

    return bottom_up_layers
