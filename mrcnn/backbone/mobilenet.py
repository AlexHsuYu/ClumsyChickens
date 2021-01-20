import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import re
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras import initializers
from keras import regularizers
from keras import constraints
#MobilenetV1 Imports
import keras.initializers as KI
import keras.regularizers as KR
import keras.constraints as KC
from keras.utils import conv_utils
from keras.engine import InputSpec
import utils
############################################################
#  MobileNetV2 Graph
############################################################

# Code adpated from:
# https://github.com/xiaochus/MobileNetV2/blob/master/mobilenet_v2.py

def fullmatch(regex, string, flags=0):
    """Emulate python-3.4 re.fullmatch()."""
#    return fullmatch("(?:" + regex + r")\Z", string, flags=flags)
    m = re.match(regex, string, flags=flags)
    if m and m.span()[1] == len(string):
        return m

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.
    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)

def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    # Currently supports ResNet and Mobilenet
    assert config.BACKBONE in ["resnet50", "resnet101", "mobilenetv1", "mobilenetv2"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True,train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]

############################################################
#  Mobilenet V1 Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/mobilenet.py

def relu6(x):
    return K.relu(x, max_value=6)

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

class DepthwiseConv2D(KL.Conv2D):
    """Depthwise separable 2D convolution.
    Depthwise Separable convolutions consists in performing
    just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.
    # Arguments
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](keras./activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](keras./initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](keras./initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](keras./regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](keras./regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](keras./regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](keras./constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](keras./constraints.md)).
    # Input shape
        4D tensor with shape:
        `[batch, channels, rows, cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, rows, cols, channels]` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1),block_id=1, train_bn=False):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = KL.Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv{}'.format(block_id))(inputs)
    x = BatchNorm(axis=channel_axis, name='conv{}_bn'.format(block_id))(x, training = train_bn)
    return KL.Activation(relu6, name='conv{}_relu'.format(block_id))(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1, train_bn=False):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    # Depthwise
    x = DepthwiseConv2D((3, 3),
                    padding='same',
                    depth_multiplier=depth_multiplier,
                    strides=strides,
                    use_bias=False,
                    name='conv_dw_{}'.format(block_id))(inputs)
    x = BatchNorm(axis=channel_axis, name='conv_dw_{}_bn'.format(block_id))(x, training=train_bn)
    x = KL.Activation(relu6, name='conv_dw_{}_relu'.format(block_id))(x)
    # Pointwise
    x = KL.Conv2D(pointwise_conv_filters, (1, 1),
                    padding='same',
                    use_bias=False,
                    strides=(1, 1),
                    name='conv_pw_{}'.format(block_id))(x)
    x = BatchNorm(axis=channel_axis, name='conv_pw_{}_bn'.format(block_id))(x, training=train_bn)
    return KL.Activation(relu6, name='conv_pw_{}_relu'.format(block_id))(x)


def mobilenetv1_graph(inputs, architecture, alpha=1.0, depth_multiplier=1, train_bn = False):
    """MobileNetv1
    This function defines a MobileNetv1 architectures.
    # Arguments
        inputs: Inuput Tensor, e.g. an image
        architecture: to preserve consistency
        alpha: Width Multiplier
        depth_multiplier: Resolution Multiplier
        train_bn: Boolean. Train or freeze Batch Norm layres
    # Returns
        five MobileNetv1 model stages.
    """
    assert architecture in ["mobilenetv1"]
    # Stage 1
    x      = _conv_block(inputs, 32, alpha, strides=(2, 2), block_id=0, train_bn=train_bn)              #Input Resolution: 224 x 224
    C1 = x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, train_bn=train_bn)       #Input Resolution: 112 x 112

    # Stage 2
    x      = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2, train_bn=train_bn)
    C2 = x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3, train_bn=train_bn)      #Input Resolution: 56 x 56

    # Stage 3
    x      = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4, train_bn=train_bn)
    C3 = x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5, train_bn=train_bn)      #Input Resolution: 28 x 28

    # Stage 4
    x      = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6, train_bn=train_bn)
    x      = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, train_bn=train_bn)
    x      = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, train_bn=train_bn)
    x      = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, train_bn=train_bn)
    x      = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, train_bn=train_bn)
    C4 = x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, train_bn=train_bn)     #Input Resolution: 14 x 14

    # Stage 5
    x      = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12, train_bn=train_bn)
    C5 = x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13, train_bn=train_bn)    #Input Resolution: 7x7
    return [C1, C2, C3, C4, C5]


############################################################
#  MobileNetV2 Graph
############################################################

# Code adpated from:
# https://github.com/xiaochus/MobileNetV2/blob/master/mobilenet_v2.py

"""MobileNet v2 models for Keras.
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

def _bottleneck(inputs, filters, kernel, t, s, r=False, alpha=1.0, block_id=1, train_bn = False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    filters = int(alpha * filters)

    x = _conv_block(inputs, tchannel, alpha, (1, 1), (1, 1),block_id=block_id,train_bn=train_bn)

    x = DepthwiseConv2D(kernel,
                    strides=(s, s),
                    depth_multiplier=1,
                    padding='same',
                    name='conv_dw_{}'.format(block_id))(x)
    x = BatchNorm(axis=channel_axis,name='conv_dw_{}_bn'.format(block_id))(x, training=train_bn)
    x = KL.Activation(relu6, name='conv_dw_{}_relu'.format(block_id))(x)

    x = KL.Conv2D(filters,
                    (1, 1),
                    strides=(1, 1),
                    padding='same',
                    name='conv_pw_{}'.format(block_id))(x)
    x = BatchNorm(axis=channel_axis, name='conv_pw_{}_bn'.format(block_id))(x, training=train_bn)

    if r:
        x = KL.add([x, inputs], name='res{}'.format(block_id))
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n, alpha, block_id, train_bn):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides, False, alpha, block_id, train_bn)

    for i in range(1, n):
        block_id += 1
        x = _bottleneck(x, filters, kernel, t, 1, True, alpha, block_id, train_bn)

    return x


def mobilenetv2_graph(inputs, architecture, alpha = 1.0, train_bn = False):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        inputs: Inuput Tensor, e.g. an image
        architecture: to preserve consistency
        alpha: Width Multiplier
        train_bn: Boolean. Train or freeze Batch Norm layres
    # Returns
        five MobileNetv2 model stages.
    """
    assert architecture in ["mobilenetv2"]

    x      = _conv_block(inputs, 32, alpha, (3, 3), strides=(2, 2), block_id=0, train_bn=train_bn)                      # Input Res: 1
    C1 = x = _inverted_residual_block(x, 16,  (3, 3), t=1, strides=1, n=1, alpha=1.0, block_id=1, train_bn=train_bn)	# Input Res: 1/2
    C2 = x = _inverted_residual_block(x, 24,  (3, 3), t=6, strides=2, n=2, alpha=1.0, block_id=2, train_bn=train_bn)	# Input Res: 1/2
    C3 = x = _inverted_residual_block(x, 32,  (3, 3), t=6, strides=2, n=3, alpha=1.0, block_id=4, train_bn=train_bn)	# Input Res: 1/4
    x      = _inverted_residual_block(x, 64,  (3, 3), t=6, strides=2, n=4, alpha=1.0, block_id=7, train_bn=train_bn)	# Input Res: 1/8
    C4 = x = _inverted_residual_block(x, 96,  (3, 3), t=6, strides=1, n=3, alpha=1.0, block_id=11, train_bn=train_bn)	# Input Res: 1/8
    x      = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3, alpha=1.0, block_id=14, train_bn=train_bn)	# Input Res: 1/16
    C5 = x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1, alpha=1.0, block_id=17, train_bn=train_bn)	# Input Res: 1/32
    #x = _conv_block(x, 1280, alpha, (1, 1), strides=(1, 1), block_id=18, train_bn=train_bn)                            # Input Res: 1/32

    return [C1,C2,C3,C4,C5]
