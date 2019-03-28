from __future__ import division

import os
import six
import h5py
import numpy as np
import cv2
import math
import scipy.spatial
import imageio

from scipy import (
    interpolate,
    ndimage
)

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    UpSampling2D
)
from keras.layers.convolutional import (
    Conv1D,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import (
    add,
    Add,
    Concatenate
)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import (
    Lambda,
    Reshape
)
from keras.layers.advanced_activations import LeakyReLU 
from keras.regularizers import l2
from keras.callbacks import Callback
from keras import backend as K

from image2 import (
    array_to_img,
    load_img2
)
from medpy.metric.binary import (
    hd,
    dc
)

import tensorflow as tf


try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

'''
def bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)
'''

def scalar_multiplication(**params):
    """A layer to multiply a tensor by a scalar
    """
    name = params.setdefault("name", "")
    scalar = params.setdefault("scalar", 1.0)
    def f(input):
        return Lambda(lambda x: tf.scalar_mul(scalar, x))(input)
    return f

def tensor_slice(dimension, start, end):
    def f(input):
        if dimension == 0:
            return input[start:end]
        if dimension == 1:
            return input[:, start:end]
        if dimension == 2:
            return input[:, :, start:end]
        if dimension == 3:
            return input[:, :, :, start:end]
    return Lambda(f)
    

def bn_relu(**params):
    """Helper to build a BN -> relu block
    """
    name = params.setdefault("name", "")
    name_bn = name + "_bn"
    name_relu = name + "_relu"
    def f(input):
        norm = BatchNormalization(axis=CHANNEL_AXIS,
                                  momentum=0.99,
                                  epsilon=1e-3,
                                  center=True,
                                  scale=True,
                                  beta_initializer='zeros',
                                  gamma_initializer='ones',
                                  moving_mean_initializer='zeros',
                                  moving_variance_initializer='ones',
                                  beta_regularizer=None,
                                  gamma_regularizer=None,
                                  beta_constraint=None,
                                  gamma_constraint=None,
                                  name=name_bn)(input)
        return Activation("relu", name=name_relu)(norm)

    return f


def bn_leakyrelu(**params):
    """Helper to build a BN -> leaky relu block
    """
    name = params.setdefault("name", "")
    alpha = params.setdefault("alpha", 0.1)
    name_bn = name + "_bn"
    name_relu = name + "_relu"
    def f(input):
        norm = BatchNormalization(axis=CHANNEL_AXIS,
                                  momentum=0.99,
                                  epsilon=1e-3,
                                  center=True,
                                  scale=True,
                                  beta_initializer='zeros',
                                  gamma_initializer='ones',
                                  moving_mean_initializer='zeros',
                                  moving_variance_initializer='ones',
                                  beta_regularizer=None,
                                  gamma_regularizer=None,
                                  beta_constraint=None,
                                  gamma_constraint=None,
                                  name=name_bn)(input)
        return LeakyReLU(alpha=alpha, name=name_relu)(norm)

    return f


def conv1D_relu(**conv_params):
    """Helper to build a conv1D -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "glorot_uniform")
    padding = conv_params.setdefault("padding", "valid")
    #kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "conv")
    name_relu = name + "_relu"

    def f(input):
        conv = Conv1D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides, 
                      padding=padding,
                      dilation_rate=1,
                      activation=None,
                      use_bias=True,
                      kernel_initializer=kernel_initializer,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name=name)(input)
        return Activation("relu", name=name_relu)(conv)

    return f



def conv1D_bn_relu(**conv_params):
    """Helper to build a conv1D -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "glorot_uniform")
    padding = conv_params.setdefault("padding", "valid")
    #kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "conv")
    name_relu = name + "_relu"

    def f(input):
        conv = Conv1D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides, 
                      padding=padding,
                      dilation_rate=1,
                      activation=None,
                      use_bias=True,
                      kernel_initializer=kernel_initializer,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name=name)(input)
        return bn_relu(name=name)(conv)

    return f





def conv_relu(**conv_params):
    """Helper to build a conv -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    #kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "conv")
    name_relu = name + "_relu"

    def f(input):
        conv = Conv2D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides, 
                      padding=padding,
                      data_format=None,
                      dilation_rate=(1, 1),
                      activation=None,
                      use_bias=True,
                      kernel_initializer=kernel_initializer,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name=name)(input)
        return Activation("relu", name=name_relu)(conv)

    return f



def conv_leakyrelu(**conv_params):
    """Helper to build a conv -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    #kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    alpha = conv_params.setdefault("alpha", 0.1)
    name = conv_params.setdefault("name", "conv")
    name_relu = name + "_relu"

    def f(input):
        conv = Conv2D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides, 
                      padding=padding,
                      data_format=None,
                      dilation_rate=(1, 1),
                      activation=None,
                      use_bias=True,
                      kernel_initializer=kernel_initializer,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name=name)(input)
        return LeakyReLU(alpha=alpha, name=name_relu)(conv)

    return f



def conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    #kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "conv")

    def f(input):
        conv = Conv2D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides, 
                      padding=padding,
                      data_format=None,
                      dilation_rate=(1, 1),
                      activation=None,
                      use_bias=True,
                      kernel_initializer=kernel_initializer,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name=name)(input)
        return bn_relu(name=name)(conv)

    return f


def conv_bn_leakyrelu(**conv_params):
    """Helper to build a conv -> BN -> leaky relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    #kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    alpha = conv_params.setdefault("alpha", 0.1)
    name = conv_params.setdefault("name", "conv")

    def f(input):
        conv = Conv2D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides, 
                      padding=padding,
                      data_format=None,
                      dilation_rate=(1, 1),
                      activation=None,
                      use_bias=True,
                      kernel_initializer=kernel_initializer,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name=name)(input)
        return bn_relu(alpha=alpha, name=name)(conv)

    return f


def deconv_relu(**deconv_params):
    """Helper to build a deconv -> relu block
    """
    filters = deconv_params["filters"]
    kernel_size = deconv_params.setdefault("kernel_size", (2, 2))
    strides = deconv_params.setdefault("strides", (2, 2))
    kernel_initializer = deconv_params.setdefault("kernel_initializer", "he_normal")
    padding = deconv_params.setdefault("padding", "valid")
    #kernel_regularizer = deconv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = deconv_params.setdefault("kernel_regularizer", None)
    name = deconv_params.setdefault("name", "deconv")
    name_relu = name + "_relu"

    def f(input):
        deconv = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name)(input)
        return Activation("relu", name=name_relu)(deconv)

    return f



def deconv_bn_relu(**deconv_params):
    """Helper to build a deconv -> BN -> relu block
    """
    filters = deconv_params["filters"]
    kernel_size = deconv_params.setdefault("kernel_size", (2, 2))
    strides = deconv_params.setdefault("strides", (2, 2))
    kernel_initializer = deconv_params.setdefault("kernel_initializer", "he_normal")
    padding = deconv_params.setdefault("padding", "valid")
    #kernel_regularizer = deconv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = deconv_params.setdefault("kernel_regularizer", None)
    name = deconv_params.setdefault("name", "deconv")

    def f(input):
        deconv = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name)(input)
        return bn_relu(name=name)(deconv)

    return f


def bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in "Identity Mappings in Deep Residual         
    Networks" (http://arxiv.org/pdf/1603.05027v2.pdf)
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)

    def f(input):
        activation = bn_relu()(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(activation)

    return f


def dense_leakyrelu(**params):
    """Helper to build a dense (fully connected) -> BN -> leaky relu block
    """
    units = params["units"]
    alpha = params.setdefault("alpha", 0.1)
    kernel_initializer = params.setdefault("kernel_initializer", "glorot_uniform")
    kernel_regularizer = params.setdefault("kernel_regularizer", None)
    name = params.setdefault("name", "dense")
    name_leakyrelu = name + "_leakyrelu"

    def f(input):
        den = Dense(units=units, 
                    activation=None,
                    use_bias=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer='zeros',
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None, 
                    bias_constraint=None,
                    name=name)(input)
        return LeakyReLU(alpha=alpha, name=name_leakyrelu)(den)

    return f




def shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
            kernel_size=(1, 1),
            strides=(stride_width, stride_height),
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in "Identity Mappings in Deep Residual         
    Networks" (http://arxiv.org/pdf/1603.05027v2.pdf)
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                strides=init_strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3),
                strides=init_strides)(input)

        residual = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in "Identity Mappings in Deep Residual         
    Networks" (http://arxiv.org/pdf/1603.05027v2.pdf)
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                strides=init_strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = bn_relu_conv(filters=filters, kernel_size=(1, 1),
                strides=init_strides)(input)

        conv_3_3 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return shortcut(input, residual)

    return f

def conv_conv_relu(**conv_params):
    """Helper to build a conv-conv-relu network block
    """
    filters = conv_params["filters"]
    kernel_size1 = conv_params["kernel_size1"]
    kernel_size2 = conv_params["kernel_size2"]
    strides1 = conv_params.setdefault("strides1", (1, 1))
    strides2 = conv_params.setdefault("strides2", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "ccr")

    def f(input):
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size1,
            strides=strides1, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1")(input)
        conv2 = Conv2D(filters=filters, kernel_size=kernel_size2,
            strides=strides2, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2")(conv1)

        return Activation("relu", name=name+"_relu")(conv2)

    return f

def conv_conv_bn_relu(**conv_params):
    """Helper to build a conv-conv-bn-relu network block
    """
    filters = conv_params["filters"]
    kernel_size1 = conv_params["kernel_size1"]
    kernel_size2 = conv_params["kernel_size2"]
    strides1 = conv_params.setdefault("strides1", (1, 1))
    strides2 = conv_params.setdefault("strides2", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "ccbr")

    def f(input):
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size1,
            strides=strides1, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1")(input)
        conv2 = Conv2D(filters=filters, kernel_size=kernel_size2,
            strides=strides2, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2")(conv1)

        return bn_relu(name=name)(conv2)

    return f


def deconv_deconv_relu(**deconv_params):
    """Helper to build a deconv-deconv-relu block
    """
    filters = deconv_params["filters"]
    kernel_size1 = deconv_params["kernel_size1"]
    kernel_size2 = deconv_params["kernel_size2"]
    strides1 = deconv_params["strides1"]
    strides2 = deconv_params["strides2"]
    kernel_initializer = deconv_params.setdefault("kernel_initializer", "he_normal")
    padding = deconv_params.setdefault("padding", "same")
    #kernel_regularizer = deconv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = deconv_params.setdefault("kernel_regularizer", None)
    name = deconv_params.setdefault("name", "ddbr")

    def f(input):
        deconv1 = Conv2DTranspose(filters=filters, kernel_size=kernel_size1,
            strides=strides1, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_deconv1")(input)
        deconv2 = Conv2DTranspose(filters=filters, kernel_size=kernel_size2,
            strides=strides2, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_deconv2")(deconv1)
        return Activation("relu", name=name+"_relu")(deconv2)

    return f


def deconv_deconv_bn_relu(**deconv_params):
    """Helper to build a deconv-deconv-bn-relu block
    """
    filters = deconv_params["filters"]
    kernel_size1 = deconv_params["kernel_size1"]
    kernel_size2 = deconv_params["kernel_size2"]
    strides1 = deconv_params["strides1"]
    strides2 = deconv_params["strides2"]
    kernel_initializer = deconv_params.setdefault("kernel_initializer", "he_normal")
    padding = deconv_params.setdefault("padding", "same")
    #kernel_regularizer = deconv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = deconv_params.setdefault("kernel_regularizer", None)
    name = deconv_params.setdefault("name", "ddbr")

    def f(input):
        deconv1 = Conv2DTranspose(filters=filters, kernel_size=kernel_size1,
            strides=strides1, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_deconv1")(input)
        deconv2 = Conv2DTranspose(filters=filters, kernel_size=kernel_size2,
            strides=strides2, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_deconv2")(deconv1)
        return bn_relu(name=name)(deconv2)

    return f






def gcn_block(**conv_params):
    """Helper to build a global convolutional network block
    which is proposed in "Large Kernel Matters -- Improve Semantic Segmentation by Global 
    Convolutional Network" (https://arxiv.org/pdf/1703.02719.pdf)
    """
    filters = conv_params["filters"]
    kernel_size1 = conv_params["kernel_size1"]
    kernel_size2 = conv_params["kernel_size2"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "gcn")

    def f(input):
        conv1_1 = Conv2D(filters=filters, kernel_size=kernel_size1,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1_1")(input)
        conv1_2 = Conv2D(filters=filters, kernel_size=kernel_size2,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1_2")(conv1_1)

        conv2_1 = Conv2D(filters=filters, kernel_size=kernel_size2,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2_1")(input)
        conv2_2 = Conv2D(filters=filters, kernel_size=kernel_size1,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2_2")(conv2_1)

        return add([conv1_2, conv2_2], name=name+"_add")

    return f


def boundary_refinement_block(**conv_params):
    """Helper to build a boundary refinement block
    which is proposed in "Large Kernel Matters -- Improve Semantic Segmentation by Global 
    Convolutional Network" (https://arxiv.org/pdf/1703.02719.pdf)
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "br")

    def f(input):
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1")(input)
        relu = Activation("relu", name=name+"_relu")(conv1)
        conv2 = Conv2D(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2")(relu)

        return add([input, conv2], name=name+"_add")

    return f

def conv_relu_repetition_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, name="conv_block"):
    """Builds a block with repeating convolution + relu blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and first_layer_down_size:
                init_strides = (2, 2)
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size
            input = conv_relu(filters=filters, kernel_size=kernel_size_i, 
                strides=init_strides, name=name+"_conv"+str(i))(input)
        return input

    return f


def conv_bn_relu_repetition_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, name="conv_block"):
    """Builds a block with repeating convolution + batch_normalization + relu blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and first_layer_down_size:
                init_strides = (2, 2)
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size

            input = conv_bn_relu(filters=filters, kernel_size=kernel_size_i, 
                strides=init_strides, name=name+"_conv"+str(i))(input)
        return input

    return f


def conv_bn_leakyrelu_repetition_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, alpha=0.1, name="conv_block"):
    """Builds a block with repeating convolution + batch_normalization + relu blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and first_layer_down_size:
                init_strides = (2, 2)
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size

            input = conv_bn_leakyrelu(filters=filters, kernel_size=kernel_size_i, 
                strides=init_strides, alpha=alpha, name=name+"_conv"+str(i))(input)
        return input

    return f


def conv_bn_leakyrelu_res_repetition_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, alpha=0.1, name="conv_block"):
    """Builds a block with repeating convolution + batch_normalization + relu blocks.
    """
    def f(input):
        init_strides = (1, 1)
        if first_layer_down_size:
            init_strides = (2, 2)
        if isinstance(kernel_size, list):
            kernel_size_0 = kernel_size[0]
        else:
            kernel_size_0 = kernel_size

        input = conv_bn_leakyrelu(filters=filters, kernel_size=kernel_size_0, 
            strides=init_strides, alpha=alpha, name=name+"_conv"+str(0))(input)

        for i in range(1, repetitions):
            init_strides = (1, 1)
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size

            if i == 1:
                input1 = conv_bn_leakyrelu(filters=filters, kernel_size=kernel_size_i, 
                    strides=init_strides, alpha=alpha, name=name+"_conv"+str(i))(input)
            else:
                input1 = conv_bn_leakyrelu(filters=filters, kernel_size=kernel_size_i, 
                    strides=init_strides, alpha=alpha, name=name+"_conv"+str(i))(input1)

        if repetitions > 1:
            return add([input, input1], name=name+"_add")
        else:
            return input

    return f


def conv_relu_repetition_residual_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, name="conv_block"):
    """Builds a block with repeating convolution + relu blocks and
    a residual connection.
    """
    def f(input):
        init_strides = (1, 1)
        if first_layer_down_size:
            init_strides = (2, 2)
        input = conv_relu(filters=filters, kernel_size=kernel_size, 
            strides=init_strides, name=name+"_conv"+str(0))(input)
        if repetitions == 2:
            input1 = conv_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(1))(input)
            return add([input, input1], name=name+"_add")

        if repetitions == 3:
            input1 = conv_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(1))(input)
            input2 = conv_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(2))(input1)
            return add([input, input2], name=name+"_add")

    return f


def conv_bn_relu_repetition_residual_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, name="conv_block"):
    """Builds a block with repeating convolution + batch_normalization + relu blocks and
    a residual connection.
    """
    def f(input):
        init_strides = (1, 1)
        if first_layer_down_size:
            init_strides = (2, 2)
        input = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
            strides=init_strides, name=name+"_conv"+str(0))(input)
        if repetitions == 2:
            input1 = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(1))(input)
            return add([input, input1], name=name+"_add")

        if repetitions == 3:
            input1 = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(1))(input)
            input2 = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(2))(input1)
            return add([input, input2], name=name+"_add")

    return f


def deconv_conv_relu_repetition_block(filters, kernel_size, repetitions,
        name="deconv_block"):
    """Builds a block with first deconvolution + relu, then 
    concatenate with input2, and finally repeating convolution + batch_normalization + relu 
    blocks.
    """
    def f(input, input2):
        input = deconv_relu(filters=filters, name=name+"_deconv")(input)
        input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            input = conv_relu(filters=filters, kernel_size=kernel_size, 
                name=name+"_conv"+str(i))(input)
        return input

    return f


def deconv_conv_bn_relu_repetition_block(filters, kernel_size, repetitions,
        name="deconv_block"):
    """Builds a block with first deconvolution + batch_normalization + relu, then 
    concatenate with input2, and finally repeating convolution + batch_normalization + relu 
    blocks.
    """
    def f(input, input2):
        input = deconv_bn_relu(filters=filters, name=name+"_deconv")(input)
        input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            input = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
                name=name+"_conv"+str(i))(input)
        return input

    return f


def up_conv_relu_repetition_block(filters, kernel_size, repetitions,
        name="up_conv_block"):
    """Builds a block with first upsampling, then 
    concatenate with input2, and finally repeating convolution + relu 
    blocks.
    """
    def f(*args):
        input = args[0]
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        if len(args) > 1:
            input2 = args[1]
            input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size
            input = conv_relu(filters=filters, kernel_size=kernel_size_i,
                name=name+"_conv"+str(i))(input)
        return input

    return f

def up_conv_bn_relu_repetition_block(filters, kernel_size, repetitions,
        name="up_conv_block"):
    """Builds a block with first upsampling, then 
    concatenate with input2, and finally repeating convolution + batch_normalization + relu 
    blocks.
    """
    def f(*args):
        input = args[0]
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        if len(args) > 1:
            input2 = args[1]
            input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size
            input = conv_bn_relu(filters=filters, kernel_size=kernel_size_i,
                name=name+"_conv"+str(i))(input)
        return input

    return f


def up_conv_relu_repetition_block2(filters, kernel_size, repetitions, up_filters_multiple=2,
        name="up_conv_block"):
    """Builds a block with first upsampling, then convolution + relu,
    concatenate with input2, and finally repeating convolution + relu 
    blocks.
    """
    def f(input, input2):
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        input = conv_relu(filters=up_filters_multiple*filters, kernel_size=kernel_size,
            name=name+"_0th_conv")(input)
        input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            input = conv_relu(filters=filters, kernel_size=kernel_size,
                name=name+"_conv"+str(i))(input)
        return input

    return f


def up_conv_bn_relu_repetition_block2(filters, kernel_size, repetitions, up_filters_multiple=2,
        name="up_conv_block"):
    """Builds a block with first upsampling, then convolution + batch_normalization + relu,
    concatenate with input2, and finally repeating convolution + batch_normalization + relu 
    blocks.
    """
    def f(input, input2):
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        input = conv_bn_relu(filters=up_filters_multiple*filters, kernel_size=kernel_size,
            name=name+"_0th_conv")(input)
        input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            input = conv_bn_relu(filters=filters, kernel_size=kernel_size,
                name=name+"_conv"+str(i))(input)
        return input

    return f

def up_conv_relu_repetition_residual_block2(filters, kernel_size, repetitions, up_filters_multiple=1,
        name="up_conv_block"):
    """Builds a block with first upsampling, then convolution + batch_normalization + relu,
    concatenate with input2, and then repeating convolution + batch_normalization + relu 
    blocks, and finally a residual connection.
    """
    def f(input, input2):
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        input = conv_relu(filters=up_filters_multiple*filters, kernel_size=kernel_size,
            name=name+"_0th_conv")(input)
        result = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            result = conv_relu(filters=filters, kernel_size=kernel_size,
                name=name+"_conv"+str(i))(result)
        return add([input, result], name=name+"_add")

    return f

def up_conv_bn_relu_repetition_residual_block2(filters, kernel_size, repetitions, up_filters_multiple=1,
        name="up_conv_block"):
    """Builds a block with first upsampling, then convolution + batch_normalization + relu,
    concatenate with input2, and then repeating convolution + batch_normalization + relu 
    blocks, and finally a residual connection.
    """
    def f(input, input2):
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        input = conv_bn_relu(filters=up_filters_multiple*filters, kernel_size=kernel_size,
            name=name+"_0th_conv")(input)
        result = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            result = conv_bn_relu(filters=filters, kernel_size=kernel_size,
                name=name+"_conv"+str(i))(result)
        return add([input, result], name=name+"_add")

    return f


def print_hdf5_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()


def dice_coef(y_true, y_pred, smooth=0.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred, smooth=0.0):
    return -dice_coef(y_true, y_pred, smooth)


def dice_coef2(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true = tf.where(y_true > 0.5, K.ones_like(y_true), K.zeros_like(y_true))
    #y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    sum = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (sum + smooth), axis=0)


def dice_coef2_loss(y_true, y_pred, smooth=1.0):
    return -dice_coef2(y_true, y_pred, smooth)

def jaccard_coef2(y_true, y_pred, smooth=0.0):
    #y_true_f = K.flatten(y_true)
    y_true = tf.where(y_true > 0.5, K.ones_like(y_true), K.zeros_like(y_true))
    #y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    sum = K.sum(y_true * y_true, axis=[1,2,3]) + K.sum(y_pred * y_pred, axis=[1,2,3])
    return K.mean((1.0 * intersection + smooth) / (sum - intersection + smooth), axis=0)


def jaccard_coef2_loss(y_true, y_pred, smooth=0.0):
    return -jaccard_coef2(y_true, y_pred, smooth)


def dice_coef3(y_true, y_pred, smooth=0.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3, axis=[1,2,3]) + K.sum(y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return (res0 + res1 + res2 + res3) / 4.0
    #return (res1 + res2 + res3) / 3.0


def dice_coef3_loss(y_true, y_pred, smooth=0.0):
    return -dice_coef3(y_true, y_pred, smooth)


def dice_coef3_0(y_true, y_pred, smooth=1e-10):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred0 = tf.where(K.equal(y_pred, 0.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f0 = K.flatten(y_pred0)

    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    return res0


def dice_coef3_1(y_true, y_pred, smooth=1e-10):
    #y_true_f = K.flatten(y_true)
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    
    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred1 = tf.where(K.equal(y_pred, 1.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f1 = K.flatten(y_pred1)

    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    return res1


def dice_coef3_2(y_true, y_pred, smooth=1e-10):
    #y_true_f = K.flatten(y_true)
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)),
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred2 = tf.where(K.equal(y_pred, 2.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f2 = K.flatten(y_pred2)

    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    return res2

def dice_coef3_3(y_true, y_pred, smooth=1e-10):
    #y_true_f = K.flatten(y_true)
    y_pred = tf.to_float(y_pred)
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)),
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred3 = tf.where(K.equal(y_pred, 3.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f3 = K.flatten(y_pred3)

    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3, axis=[1,2,3]) + K.sum(y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return res3


def dice_coef4(y_true, y_pred, smooth=0.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0 * y_true0, axis=[1,2,3]) + K.sum(y_pred0 * y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1 * y_true1, axis=[1,2,3]) + K.sum(y_pred1 * y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2 * y_true2, axis=[1,2,3]) + K.sum(y_pred2 * y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3 * y_true3, axis=[1,2,3]) + K.sum(y_pred3 * y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return (res0 + res1 + res2 + res3) / 4.0
    #return (res1 + res2 + res3) / 3.0


def dice_coef4_loss(y_true, y_pred, smooth=0.0):
    return -dice_coef4(y_true, y_pred, smooth)


def dice_coef5(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3, axis=[1,2,3]) + K.sum(y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return (res0 + res1 + res2 + res3) / 4.0
    #return (res1 + res2 + res3) / 3.0


def dice_coef5_loss(y_true, y_pred, smooth=1.0):
    return -dice_coef5(y_true, y_pred, smooth)


def dice_coef5_0(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred0 = tf.where(K.equal(y_pred, 0.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f0 = K.flatten(y_pred0)

    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    return res0


def dice_coef5_1(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    
    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred1 = tf.where(K.equal(y_pred, 1.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f1 = K.flatten(y_pred1)

    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    return res1


def dice_coef5_2(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)),
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred2 = tf.where(K.equal(y_pred, 2.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f2 = K.flatten(y_pred2)

    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    return res2

def dice_coef5_3(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_pred = tf.to_float(y_pred)
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)),
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred3 = tf.where(K.equal(y_pred, 3.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f3 = K.flatten(y_pred3)

    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3, axis=[1,2,3]) + K.sum(y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return res3



def dice_coef6(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    return (res0 + res1 + res2) / 3.0


def dice_coef6_loss(y_true, y_pred, smooth=1.0):
    return -dice_coef6(y_true, y_pred, smooth)

def dice_coef7_array(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true = np.where(np.equal(y_true, 200.0 * np.ones_like(y_true)), 
                      np.ones_like(y_true), np.zeros_like(y_true))
    #y_pred_f = K.flatten(y_pred)
    intersection = np.sum(y_true * y_pred, axis=(1,2,3))
    sum = np.sum(y_true, axis=(1,2,3)) + np.sum(y_pred, axis=(1,2,3))
    #return np.mean((2. * intersection + smooth) / (sum + smooth), axis=0)
    return (2. * intersection + smooth) / (sum + smooth)


def dice_coef7(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true = tf.where(K.equal(y_true, 200.0 * K.ones_like(y_true)), 
                      K.ones_like(y_true), K.zeros_like(y_true))
    #y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    sum = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (sum + smooth), axis=0)


def dice_coef7_loss(y_true, y_pred, smooth=1.0):
    return -dice_coef7(y_true, y_pred, smooth)



def jaccard_coef3_1(y_true, y_pred, smooth=0.0):
    '''
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))

    shape = tf.shape(y_true)
    y_true0 = K.reshape(y_true0, (shape[0], shape[1], shape[2]))
    y_true1 = K.reshape(y_true1, (shape[0], shape[1], shape[2]))
    y_true2 = K.reshape(y_true2, (shape[0], shape[1], shape[2]))
    y_true3 = K.reshape(y_true3, (shape[0], shape[1], shape[2]))

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])

    y_pred0 = K.reshape(y_pred0, (shape[0], shape[1], shape[2]))
    y_pred1 = K.reshape(y_pred1, (shape[0], shape[1], shape[2]))
    y_pred2 = K.reshape(y_pred2, (shape[0], shape[1], shape[2]))
    y_pred3 = K.reshape(y_pred3, (shape[0], shape[1], shape[2]))

    intersection0 = tf.norm(y_true0 * y_pred0, axis=[1,2])
    sum0 =  K.sum(y_true0 * y_true0, axis=[1,2]) + K.sum(y_pred0 * y_pred0, axis=[1,2])
    res0 = (1.0 * intersection0 + smooth) / (sum0 - intersection0 + smooth)

    intersection1 = tf.norm(y_true1 * y_pred1, axis=[1,2])
    sum1 =  K.sum(y_true1 * y_true1, axis=[1,2]) + K.sum(y_pred1 * y_pred1, axis=[1,2])
    res1 = (1.0 * intersection1 + smooth) / (sum1 - intersection1 + smooth)

    intersection2 = tf.norm(y_true2 * y_pred2, axis=[1,2])
    sum2 =  K.sum(y_true2 * y_true2, axis=[1,2]) + K.sum(y_pred2 * y_pred2, axis=[1,2])
    res2 = (1.0 * intersection2 + smooth) / (sum2 - intersection2 + smooth)

    intersection3 = tf.norm(y_true3 * y_pred3, axis=[1,2])
    sum3 =  K.sum(y_true3 * y_true3, axis=[1,2]) + K.sum(y_pred3 * y_pred3, axis=[1,2])
    res3 = (1.0 * intersection3 + smooth) / (sum3 - intersection3 + smooth)
    '''


    #y_true_f = K.flatten(y_true)
    #y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
    #             K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    

    #y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    #intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    #sum0 = K.sum(y_true0 * y_true0, axis=[1,2,3]) + K.sum(y_pred0 * y_pred0, axis=[1,2,3])
    #res0 = K.mean((1.0 * intersection0 + smooth) / (sum0 - intersection0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1 * y_true1, axis=[1,2,3]) + K.sum(y_pred1 * y_pred1, axis=[1,2,3])
    res1 = K.mean((1.0 * intersection1 + smooth) / (sum1 - intersection1 + smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2 * y_true2, axis=[1,2,3]) + K.sum(y_pred2 * y_pred2, axis=[1,2,3])
    res2 = K.mean((1.0 * intersection2 + smooth) / (sum2 - intersection2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3 * y_true3, axis=[1,2,3]) + K.sum(y_pred3 * y_pred3, axis=[1,2,3])
    res3 = K.mean((1.0 * intersection3 + smooth) / (sum3 - intersection3 + smooth), axis=0)

 
    #return (res0 + res1 + res2 + res3) / 4.0
    return (res1 + res2 + res3) / 3.0



def jaccard_coef3_2(y_true, y_pred, smooth=0.0):
    
    y_true0 = K.ones_like(y_true)
    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])

    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0 * y_true0, axis=[1,2,3]) + K.sum(y_pred0 * y_pred0, axis=[1,2,3])
    res0 = K.mean((1.0 * intersection0 + smooth) / (sum0 - intersection0 + smooth), axis=0)

    return res0


def jaccard_coef3(y_true, y_pred, smooth=0.0):

    return tf.cond(tf.reduce_max(y_true) > 0., 
                   lambda: jaccard_coef3_1(y_true, y_pred, smooth),
                   lambda: jaccard_coef3_2(y_true, y_pred, smooth))


def jaccard_coef3_loss(y_true, y_pred, smooth=0.0):
    return -jaccard_coef3(y_true, y_pred, smooth)


def jaccard_coef4(y_true, y_pred, smooth=0.0):

    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0 * y_true0, axis=[1,2,3]) + K.sum(y_pred0 * y_pred0, axis=[1,2,3])
    res0 = K.mean((1.0 * intersection0 + smooth) / (sum0 - intersection0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1 * y_true1, axis=[1,2,3]) + K.sum(y_pred1 * y_pred1, axis=[1,2,3])
    res1 = K.mean((1.0 * intersection1 + smooth) / (sum1 - intersection1 + smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2 * y_true2, axis=[1,2,3]) + K.sum(y_pred2 * y_pred2, axis=[1,2,3])
    res2 = K.mean((1.0 * intersection2 + smooth) / (sum2 - intersection2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3 * y_true3, axis=[1,2,3]) + K.sum(y_pred3 * y_pred3, axis=[1,2,3])
    res3 = K.mean((1.0 * intersection3 + smooth) / (sum3 - intersection3 + smooth), axis=0)

 
    return (res0 + res1 + res2 + res3) / 4.0
    #return (res1 + res2 + res3) / 3.0


def jaccard_coef4_loss(y_true, y_pred, smooth=0.0):
    return -jaccard_coef4(y_true, y_pred, smooth)

def base_slice_euclidean_distance_loss(y_true, y_pred):
    y_pred_reduced = tf.reduce_max(y_pred, axis=[1,2,3])
    
    max_values = tf.reduce_max(y_true, axis=[1,2,3])
    
    labels = tf.where(K.equal(max_values, 3.0 * K.ones_like(max_values)), 
        K.ones_like(max_values), K.zeros_like(max_values))
    
    return tf.reduce_sum(tf.square(tf.subtract(y_pred_reduced, labels)))


def depth_softmax(matrix):
    sigmoid = lambda x: 1.0 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=2)
    return softmax_matrix


def mean_variance_normalization(array):
    mean = np.mean(array)
    std = np.std(array)
    adjusted_std = max(std, 1.0/np.sqrt(array.size))
    return (array - mean)/adjusted_std

def mean_variance_normalization2(array):

    percentile1 = np.percentile(array, 10)
    percentile2 = np.percentile(array, 90)
    array2 = array[np.logical_and(array > percentile1, array < percentile2)]
    mean = np.mean(array2)
    std = np.std(array2)
    '''
    percentile1 = np.percentile(array, 5)
    percentile2 = np.percentile(array, 95)
    array[array <= percentile1] = percentile1
    array[array >= percentile2] = percentile2
    mean = np.mean(array)
    std = np.std(array)
    '''
    adjusted_std = max(std, 1.0/np.sqrt(array.size))
    return 1.0 * (array - mean)/adjusted_std

def mean_variance_normalization3(array):
    percentile1 = np.percentile(array, 5)
    percentile2 = np.percentile(array, 95)
    array[array <= percentile1] = percentile1
    array[array >= percentile2] = percentile2

    shape = array.shape
    array = np.reshape(array, (shape[0], shape[1]))
    array = array.astype('uint16')
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    array = clahe.apply(array)
    array = array.astype(float)
    array = np.reshape(array, shape)

    mean = np.mean(array)
    std = np.std(array)
    adjusted_std = max(std, 1.0/np.sqrt(array.size))
    return 1.0 * (array - mean)/adjusted_std

def mean_variance_normalization4(array):

    array2 = array[array > array.min()]
    if (array2.size > 0):
        mean = np.mean(array2)
        std = np.std(array2)
        adjusted_std = max(std, 1.0/np.sqrt(array.size))
        return 1.0 * (array - mean)/adjusted_std
    else:
        return (array - array.min())


def mean_variance_normalization5(array):

    percentile1 = np.percentile(array, 5)
    percentile2 = np.percentile(array, 95)

    array2 = array[np.logical_and(array > percentile1, array < percentile2)]
    if (array2.size > 0):
        mean = np.mean(array2)
        std = np.std(array2)
        adjusted_std = max(std, 1.0/np.sqrt(array.size))
        return 1.0 * (array - mean)/adjusted_std
    else:
        return (array - array.min())

def elementwise_multiplication(array):
    return (0.02 * array)

def elementwise_multiplication2(array):
    array2 = 0.02 * array
    array2[array2 == 3.0] = 0.0
    return array2

def one_hot(indices, num_classes):
    res = []
    for i in range(num_classes):
        res += [tf.where(K.equal(indices, i * K.ones_like(indices)), 
                         K.ones_like(indices), K.zeros_like(indices))]
    return K.concatenate(res, axis=-1)
    

def mask_to_contour(mask):
    results = cv2.findContours(np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if cv2.__version__[:2] == '3.':
        coords = results[1]
    else:
        coords = results[0]
    #coords, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(coords) > 1:
        print('Multiple contours detected')
        lengths = []
        for coord in coords:
            lengths.append(len(coord))
        coords = [coords[np.argmax(lengths)]]
    if len(coords) > 0:
        coord = coords[0]
        coord = np.squeeze(coord, axis=(1,))
        coord = np.append(coord, coord[:1], axis=0)
    else:
        coord = np.empty(0)
    return coord

def hausdorff_distance(coord1, coord2, pixel_spacing):
    max_of_min1 = scipy.spatial.distance.directed_hausdorff(coord1, coord2)[0]
    max_of_min2 = scipy.spatial.distance.directed_hausdorff(coord2, coord1)[0]
    return max(max_of_min1, max_of_min2) * pixel_spacing


def extract_2D_mask_boundary(m):
    w = m.shape[0]
    h = m.shape[1]
    mm = m
    for wi in range(1, w-1):
        for hi in range(1, h-1):
            if m[wi, hi] > 0 and m[wi-1, hi] > 0 and m[wi+1, hi] > 0 and m[wi, hi-1] > 0 and m[wi, hi+1] > 0:
                mm[wi, hi] = 0
    return mm

def volume_Dice(p_volume, m_volume, min_v, max_v):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    Dices = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :].flatten()
        m_s = m[s, :, :].flatten()
        if (np.sum(m_s) > 0):
            Dice = 2.0 * np.sum(p_s * m_s) / (np.sum(p_s) + np.sum(m_s))
        else:
            Dice = -1
        Dices.append(Dice)
    return Dices

def volume_Dice_3D(p_volume, m_volume, min_v, max_v):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    
    return dc(m, p)

def volume_APD(p_volume, m_volume, min_v, max_v, pixel_spacing, eng):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    APDs = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :]
        m_s = m[s, :, :]
        p_s_ctr = mask_to_contour(p_s * 255)
        m_s_ctr = mask_to_contour(m_s * 255)
        if len(p_s_ctr.shape) == 2:
            try:
                APD = eng.average_perpendicular_distance(p_s_ctr[:, 0], p_s_ctr[:, 1], 
                    m_s_ctr[:, 0], m_s_ctr[:, 1], p_s.shape[0], p_s.shape[1], pixel_spacing)
            except Exception as e:
                print(e)
                APD = -2
        else:
            APD = -1
        APDs.append(APD)
    return APDs

def volume_APD2(p_volume, m_txt, min_v, max_v, pixel_spacing, to_original_x, to_original_y, eng):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    APDs = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :]
        p_s_ctr = mask_to_contour(p_s * 255)
        m_s_ctr = np.loadtxt(m_txt[s])
        if len(p_s_ctr.shape) == 2:
            p_s_ctr[:, 0] += to_original_x
            p_s_ctr[:, 1] += to_original_y
            try:
                APD = eng.average_perpendicular_distance(p_s_ctr[:, 0].tolist(), p_s_ctr[:, 1].tolist(), 
                    m_s_ctr[:, 0].tolist(), m_s_ctr[:, 1].tolist(), int(p_s.shape[0]), int(p_s.shape[1]), pixel_spacing)
            except Exception as e:
                print(e)
                APD = -2
        else:
            APD = -1
        APDs.append(APD)
    return APDs


def volume_hausdorff_distance(p_volume, m_volume, min_v, max_v, pixel_spacing, to_contours):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    HDs = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :]
        m_s = m[s, :, :]
        if (np.sum(m_s.flatten()) > 0):
            if (np.sum(p_s.flatten()) > 0):
                if to_contours:
                    p_s_ctr = mask_to_contour(p_s * 255)
                    m_s_ctr = mask_to_contour(m_s * 255)
                    HD = hausdorff_distance(p_s_ctr, m_s_ctr, pixel_spacing)
                else:
                    p_s_b = extract_2D_mask_boundary(p_s)
                    m_s_b = extract_2D_mask_boundary(m_s)
                    p_s_coord = np.argwhere(np.array(p_s_b, dtype=bool))
                    m_s_coord = np.argwhere(np.array(m_s_b, dtype=bool))
                    HD = hausdorff_distance(p_s_coord, m_s_coord, pixel_spacing)
            else:
                HD = -2
        else:
            HD = -1
        HDs.append(HD)
    return HDs


def volume_hausdorff_distance2(p_volume, m_txt, min_v, max_v, pixel_spacing, 
    to_original_x, to_original_y, to_contours):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    HDs = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :]
        m_s_ctr = np.loadtxt(m_txt[s])
        if (np.sum(p_s.flatten()) > 0):
            try:  
                if to_contours:
                    p_s_ctr = mask_to_contour(p_s * 255)
                    p_s_ctr[:, 0] += to_original_x
                    p_s_ctr[:, 1] += to_original_y
                    HD = hausdorff_distance(p_s_ctr, m_s_ctr, pixel_spacing)
                else:
                    p_s_b = extract_2D_mask_boundary(p_s)
                    p_s_coord = np.argwhere(np.array(p_s_b, dtype=bool))
                    p_s_coord[:, 0] += to_original_x
                    p_s_coord[:, 1] += to_original_y
                    HD = hausdorff_distance(p_s_coord, m_s_ctr, pixel_spacing)
            except Exception as e:
                print(e)
                HD = -2
        else:
            HD = -2
        HDs.append(HD)
    return HDs

def volume_hausdorff_distance_3D(p_volume, m_volume, min_v, max_v, pixel_spacing):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    
    return hd(m, p, pixel_spacing)



def mean_of_positive_elements(l):
    return 1.0 * sum([x for x in l if x >= 0]) / max(len([x for x in l if x >= 0]), 1)




def get_array_values_given_coordinates(img, x, y):
    """
    Utility function to get pixel value for coordinate
    arrays x and y from a  4D array image.
    Input
    -----
    - img: array of shape (B, H, W, C)
    - x: array of shape (B, H, W, 1)
    - y: array of shape (B, H, W, 1)
    Returns
    -------
    - output: array of shape (B, H, W, C)
    """
    shape = img.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = range(0, batch_size)
    batch_idx = np.reshape(batch_idx, (batch_size, 1, 1))
    b = np.tile(batch_idx, (1, height, width))

    x_r = np.reshape(x, (batch_size, height, width))
    y_r = np.reshape(y, (batch_size, height, width))

    indices = np.stack([b, y_r, x_r], 3)

    output = np.zeros_like(img)
    for s in range(batch_size):
        for h in range(height):
            for w in range(width):
                coord = indices[s, h, w]
                output[s, h, w, :] = img[coord[0], coord[1], coord[2], :]

    return output


def inversely_get_array_values_given_coordinates(img, x, y):
    """
    Utility function to get pixel value for coordinate
    arrays x and y from a  4D array image.
    Input
    -----
    - img: array of shape (B, H, W, C)
    - x: array of shape (B, H, W, 1)
    - y: array of shape (B, H, W, 1)
    Returns
    -------
    - output: array of shape (B, H, W, C)
    """
    shape = img.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = range(0, batch_size)
    batch_idx = np.reshape(batch_idx, (batch_size, 1, 1))
    b = np.tile(batch_idx, (1, height, width))

    x_r = np.reshape(x, (batch_size, height, width))
    y_r = np.reshape(y, (batch_size, height, width))

    indices = np.stack([b, y_r, x_r], 3)

    output = np.zeros_like(img)
    for s in range(batch_size):
        for h in range(height):
            for w in range(width):
                coord = indices[s, h, w]
                output[coord[0], coord[1], coord[2], :] = img[s, h, w, :]

    return output



def warp_array_according_to_flow(img, flow, mode='bilinear'):
    """
    Function to warp a 4D image array according to 
    a 4D flow array.
    Input
    -----
    - img: array of shape (B, H, W, C)
    - flow: array of shape (B, H, W, 2)
    Returns
    -------
    - output: array of shape (B, H, W, C)
    """
    shape = img.shape
    batch_size = shape[0]
    H = shape[1]
    W = shape[2]

    x,y = np.meshgrid(range(W), range(H))
    x = np.expand_dims(x,0)
    x = np.expand_dims(x,-1)

    y = np.expand_dims(y,0)
    y = np.expand_dims(y,-1)

    x = x.astype(float)
    y = y.astype(float)
    grid  = np.concatenate((x,y),axis = -1)
    # print grid.shape
    coords = grid+flow

    max_y = int(H - 1)
    max_x = int(W - 1)

    coords_x = coords[:, :, :, 0:1]
    coords_y = coords[:, :, :, 1:2]
    
    if mode == 'bilinear':
        x0 = coords_x
        y0 = coords_y
        x0 = (np.floor(x0)).astype(int)
        x1 = x0 + 1
        y0 = (np.floor(y0)).astype(int)
        y1 = y0 + 1

        # clip to range [0, H/W] to not violate img boundaries
        x0c = np.clip(x0, 0, max_x)
        x1c = np.clip(x1, 0, max_x)
        y0c = np.clip(y0, 0, max_y)
        y1c = np.clip(y1, 0, max_y)

        # get pixel value at corner coords
        Ia = get_array_values_given_coordinates(img, x0c, y0c)
        Ib = get_array_values_given_coordinates(img, x0c, y1c)
        Ic = get_array_values_given_coordinates(img, x1c, y0c)
        Id = get_array_values_given_coordinates(img, x1c, y1c)

        # recast as float for delta calculation
        x0 = x0.astype(float)
        x1 = x1.astype(float)
        y0 = y0.astype(float)
        y1 = y1.astype(float)

        # calculate deltas
        wa = (x1-coords_x) * (y1-coords_y)
        wb = (x1-coords_x) * (coords_y-y0)
        wc = (coords_x-x0) * (y1-coords_y)
        wd = (coords_x-x0) * (coords_y-y0)

        # compute output
        out = wa*Ia + wb*Ib + wc*Ic + wd*Id
        return out

    elif mode == 'nearest':
        x0 = (np.rint(coords_x)).astype(int)
        y0 = (np.rint(coords_y)).astype(int)

        # clip to range [0, H/W] to not violate img boundaries
        x0 = np.clip(x0, 0, max_x)
        y0 = np.clip(y0, 0, max_y)

        # get pixel value at corner coords
        out = get_array_values_given_coordinates(img, x0, y0)
        return out


def inversely_warp_array_according_to_flow(img, flow, mode='nearest'):
    """
    Function to warp a 4D image array according to 
    a 4D flow array.
    Input
    -----
    - img: array of shape (B, H, W, C)
    - flow: array of shape (B, H, W, 2)
    Returns
    -------
    - output: array of shape (B, H, W, C)
    """
    shape = img.shape
    batch_size = shape[0]
    H = shape[1]
    W = shape[2]

    x,y = np.meshgrid(range(W), range(H))
    x = np.expand_dims(x,0)
    x = np.expand_dims(x,-1)

    y = np.expand_dims(y,0)
    y = np.expand_dims(y,-1)

    x = x.astype(float)
    y = y.astype(float)
    grid  = np.concatenate((x,y),axis = -1)
    # print grid.shape
    coords = grid+flow

    max_y = int(H - 1)
    max_x = int(W - 1)

    coords_x = coords[:, :, :, 0:1]
    coords_y = coords[:, :, :, 1:2]
    
    if mode == 'nearest':
        x0 = (np.rint(coords_x)).astype(int)
        y0 = (np.rint(coords_y)).astype(int)

        # clip to range [0, H/W] to not violate img boundaries
        x0 = np.clip(x0, 0, max_x)
        y0 = np.clip(y0, 0, max_y)

        # get pixel value at corner coords
        out = inversely_get_array_values_given_coordinates(img, x0, y0)
        return out



def flow_array_diffeomorphism_loss(y_true, y_pred):
    flow = y_pred

    shape = flow.shape
    batch_size = shape[0]
    H = shape[1]
    W = shape[2]

    x,y = np.meshgrid(range(W), range(H))
    x = np.expand_dims(x,0)
    x = np.expand_dims(x,-1)

    y = np.expand_dims(y,0)
    y = np.expand_dims(y,-1)

    x = x.astype(float)
    y = y.astype(float)
    grid  = np.concatenate((x,y),axis = -1)
    # print grid.shape
    coords = grid+flow
    
    zero = tf.zeros([], dtype=tf.int32)

    coords_x = coords[:, :, :, 0:1] 
    coords_y = coords[:, :, :, 1:2] 

    coords_x0 = coords_x[:, :, 0:(W-1), :]  
    coords_x1 = coords_x[:, :, 1:W, :]      
    coords_y0 = coords_y[:, 0:(H-1), :, :]  
    coords_y1 = coords_y[:, 1:H, :, :]      

    zeros_x = np.zeros_like(coords_x0)
    zeros_y = np.zeros_like(coords_y0)

    x_adjacent_diff_clip = np.minimum(coords_x1 - coords_x0, zeros_x)
    y_adjacent_diff_clip = np.minimum(coords_y1 - coords_y0, zeros_y)

    return np.mean(np.sum(x_adjacent_diff_clip**2, axis=(1,2,3)) + np.sum(y_adjacent_diff_clip**2, axis=(1,2,3)), axis=0)



def get_tensor_values_given_coordinates(img, x, y):
    """
    Utility function to get pixel value for coordinate
    tensors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: tensor of shape (B, H, W, 1)
    - y: tensor of shape (B, H, W, 1)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(img)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    x_r = tf.reshape(x, (batch_size, height, width))
    y_r = tf.reshape(y, (batch_size, height, width))

    indices = tf.stack([b, y_r, x_r], 3)

    return tf.gather_nd(img, indices)


def warp_tensor_according_to_flow(img, flow):
    """
    Function to warp a 4D image tensor according to 
    a 4D flow tensor.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - flow: tensor of shape (B, H, W, 2)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(img)
    batch_size = shape[0]
    H = shape[1]
    W = shape[2]

    x,y = tf.meshgrid(tf.range(W), tf.range(H))
    x = tf.expand_dims(x,0)
    x = tf.expand_dims(x,-1)

    y  =tf.expand_dims(y,0)
    y = tf.expand_dims(y,-1)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    grid  = tf.concat([x,y],axis = -1)
    # print grid.shape
    coords = grid+flow

    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    coords_x = tf.slice(coords, [0,0,0,0], [-1,-1,-1,1])
    coords_y = tf.slice(coords, [0,0,0,1], [-1,-1,-1,1])
    
    x0 = coords_x
    y0 = coords_y
    x0 = tf.cast(tf.floor(x0), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y0),  tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0c = tf.clip_by_value(x0, zero, max_x)
    x1c = tf.clip_by_value(x1, zero, max_x)
    y0c = tf.clip_by_value(y0, zero, max_y)
    y1c = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_tensor_values_given_coordinates(img, x0c, y0c)
    Ib = get_tensor_values_given_coordinates(img, x0c, y1c)
    Ic = get_tensor_values_given_coordinates(img, x1c, y0c)
    Id = get_tensor_values_given_coordinates(img, x1c, y1c)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)


    # calculate deltas
    wa = (x1-coords_x) * (y1-coords_y)
    wb = (x1-coords_x) * (coords_y-y0)
    wc = (coords_x-x0) * (y1-coords_y)
    wd = (coords_x-x0) * (coords_y-y0)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out

def flow_warped_img_comparison_loss(y_true, y_pred):
    # y_true consists of 2 batch images and 2 batch ground-truths
    img0 = tf.slice(y_true, [0,0,0,0], [-1,-1,-1,1])
    img1 = tf.slice(y_true, [0,0,0,1], [-1,-1,-1,1])
    
    flow = y_pred
    warped_img0 = warp_tensor_according_to_flow(img0, flow)
    diff = warped_img0 - img1
    euclidean_dist = K.sum(diff**2, axis=[1,2,3])
    return K.mean(euclidean_dist, axis=0)

def flow_warped_gt_comparison_loss(y_true, y_pred):
    # y_true consists of 2 batch images and 2 batch ground-truths
    gt0 = tf.slice(y_true, [0,0,0,2], [-1,-1,-1,1])
    gt1 = tf.slice(y_true, [0,0,0,3], [-1,-1,-1,1])

    gt0_1 = tf.where(K.equal(gt0, 1.0 * K.ones_like(gt0)), 
                     K.ones_like(gt0), K.zeros_like(gt0))
    gt0_2 = tf.where(K.equal(gt0, 2.0 * K.ones_like(gt0)), 
                     K.ones_like(gt0), K.zeros_like(gt0))
    gt0_3 = tf.where(K.equal(gt0, 3.0 * K.ones_like(gt0)), 
                     K.ones_like(gt0), K.zeros_like(gt0))

    gt1_1 = tf.where(K.equal(gt1, 1.0 * K.ones_like(gt1)), 
                     K.ones_like(gt1), K.zeros_like(gt1))
    gt1_2 = tf.where(K.equal(gt1, 2.0 * K.ones_like(gt1)), 
                     K.ones_like(gt1), K.zeros_like(gt1))
    gt1_3 = tf.where(K.equal(gt1, 3.0 * K.ones_like(gt1)), 
                     K.ones_like(gt1), K.zeros_like(gt1))

    flow = y_pred

    warped_gt0_1 = warp_tensor_according_to_flow(gt0_1, flow)
    warped_gt0_2 = warp_tensor_according_to_flow(gt0_2, flow)
    warped_gt0_3 = warp_tensor_according_to_flow(gt0_3, flow)

    diff1 = warped_gt0_1 - gt1_1
    diff2 = warped_gt0_2 - gt1_2
    diff3 = warped_gt0_3 - gt1_3
    euclidean_dist = K.sum(diff1**2 + diff2**2 + diff3**2, axis=[1,2,3])
    return K.mean(euclidean_dist, axis=0)


def flow_warped_gt_comparison_dice_loss(y_true, y_pred):
    # y_true consists of 2 batch images and 2 batch ground-truths
    gt0 = tf.slice(y_true, [0,0,0,2], [-1,-1,-1,1])
    gt1 = tf.slice(y_true, [0,0,0,3], [-1,-1,-1,1])

    gt0_1 = tf.where(K.equal(gt0, 1.0 * K.ones_like(gt0)), 
                     K.ones_like(gt0), K.zeros_like(gt0))
    gt0_2 = tf.where(K.equal(gt0, 2.0 * K.ones_like(gt0)), 
                     K.ones_like(gt0), K.zeros_like(gt0))
    gt0_3 = tf.where(K.equal(gt0, 3.0 * K.ones_like(gt0)), 
                     K.ones_like(gt0), K.zeros_like(gt0))

    gt1_1 = tf.where(K.equal(gt1, 1.0 * K.ones_like(gt1)), 
                     K.ones_like(gt1), K.zeros_like(gt1))
    gt1_2 = tf.where(K.equal(gt1, 2.0 * K.ones_like(gt1)), 
                     K.ones_like(gt1), K.zeros_like(gt1))
    gt1_3 = tf.where(K.equal(gt1, 3.0 * K.ones_like(gt1)), 
                     K.ones_like(gt1), K.zeros_like(gt1))

    flow = y_pred

    warped_gt0_1 = warp_tensor_according_to_flow(gt0_1, flow)
    warped_gt0_2 = warp_tensor_according_to_flow(gt0_2, flow)
    warped_gt0_3 = warp_tensor_according_to_flow(gt0_3, flow)

    loss1 = dice_coef2_loss(gt1_1, warped_gt0_1, smooth=1.0)
    loss2 = dice_coef2_loss(gt1_2, warped_gt0_2, smooth=1.0)
    loss3 = dice_coef2_loss(gt1_3, warped_gt0_3, smooth=1.0)

    return loss1 + loss2 + loss3


def flow_warped_gt_comparison_dice_loss_lvc(y_true, y_pred):
    # y_true consists of 2 batch images and 2 batch ground-truths
    gt0 = tf.slice(y_true, [0,0,0,2], [-1,-1,-1,1])
    gt1 = tf.slice(y_true, [0,0,0,3], [-1,-1,-1,1])

    gt0_1 = tf.where(K.equal(gt0, 1.0 * K.ones_like(gt0)), 
                     K.ones_like(gt0), K.zeros_like(gt0))

    gt1_1 = tf.where(K.equal(gt1, 1.0 * K.ones_like(gt1)), 
                     K.ones_like(gt1), K.zeros_like(gt1))
    
    flow = y_pred

    warped_gt0_1 = warp_tensor_according_to_flow(gt0_1, flow)
    
    loss1 = dice_coef2_loss(gt1_1, warped_gt0_1, smooth=0.0)

    return loss1

def flow_warped_gt_comparison_dice_loss_lvm(y_true, y_pred):
    # y_true consists of 2 batch images and 2 batch ground-truths
    gt0 = tf.slice(y_true, [0,0,0,2], [-1,-1,-1,1])
    gt1 = tf.slice(y_true, [0,0,0,3], [-1,-1,-1,1])

    gt0_2 = tf.where(K.equal(gt0, 2.0 * K.ones_like(gt0)), 
                     K.ones_like(gt0), K.zeros_like(gt0))

    gt1_2 = tf.where(K.equal(gt1, 2.0 * K.ones_like(gt1)), 
                     K.ones_like(gt1), K.zeros_like(gt1))
    
    flow = y_pred

    warped_gt0_2 = warp_tensor_according_to_flow(gt0_2, flow)
    
    loss2 = dice_coef2_loss(gt1_2, warped_gt0_2, smooth=0.0)

    return loss2

def flow_warped_gt_comparison_dice_loss_rvc(y_true, y_pred):
    # y_true consists of 2 batch images and 2 batch ground-truths
    gt0 = tf.slice(y_true, [0,0,0,2], [-1,-1,-1,1])
    gt1 = tf.slice(y_true, [0,0,0,3], [-1,-1,-1,1])

    gt0_3 = tf.where(K.equal(gt0, 3.0 * K.ones_like(gt0)), 
                     K.ones_like(gt0), K.zeros_like(gt0))

    gt1_3 = tf.where(K.equal(gt1, 3.0 * K.ones_like(gt1)), 
                     K.ones_like(gt1), K.zeros_like(gt1))
    
    flow = y_pred

    warped_gt0_3 = warp_tensor_according_to_flow(gt0_3, flow)
    
    loss3 = dice_coef2_loss(gt1_3, warped_gt0_3, smooth=0.0)

    return loss3


def flow_diffeomorphism_loss(y_true, y_pred):
    flow = y_pred

    shape = tf.shape(flow)
    batch_size = shape[0]
    H = shape[1]
    W = shape[2]

    x,y = tf.meshgrid(tf.range(W), tf.range(H))
    x = tf.expand_dims(x,0)
    x = tf.expand_dims(x,-1)

    y  =tf.expand_dims(y,0)
    y = tf.expand_dims(y,-1)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    grid  = tf.concat([x,y],axis = -1)
    # print grid.shape
    coords = grid+flow
    
    zero = tf.zeros([], dtype=tf.int32)

    coords_x = tf.slice(coords, [0,0,0,0], [-1,-1,-1,1])
    coords_y = tf.slice(coords, [0,0,0,1], [-1,-1,-1,1])

    coords_x0 = tf.slice(coords_x, [0,0,0,0], [-1,-1,W-1,-1])
    coords_x1 = tf.slice(coords_x, [0,0,1,0], [-1,-1,W-1,-1])
    coords_y0 = tf.slice(coords_y, [0,0,0,0], [-1,H-1,-1,-1])
    coords_y1 = tf.slice(coords_y, [0,1,0,0], [-1,H-1,-1,-1])

    zeros_x = K.zeros_like(coords_x0)
    zeros_y = K.zeros_like(coords_y0)

    x_adjacent_diff_clip = tf.minimum(coords_x1 - coords_x0, zeros_x)
    y_adjacent_diff_clip = tf.minimum(coords_y1 - coords_y0, zeros_y)

    return K.mean(K.sum(x_adjacent_diff_clip**2, axis=[1,2,3]) + K.sum(y_adjacent_diff_clip**2, axis=[1,2,3]), axis=0)


def flow_combined_loss(y_true, y_pred, coeff = 1000.0):
    return flow_warped_img_comparison_loss(y_true, y_pred) + \
        coeff * flow_diffeomorphism_loss(y_true, y_pred)

def flow_combined_loss2(y_true, y_pred, coeff1 = 10.0, coeff2 = 1000.0):
    return flow_warped_img_comparison_loss(y_true, y_pred) + \
        coeff1 * flow_warped_gt_comparison_loss(y_true, y_pred) + \
        coeff2 * flow_diffeomorphism_loss(y_true, y_pred)

def flow_combined_loss3(y_true, y_pred, coeff1 = 100000.0, coeff2 = 1000.0):
    return flow_warped_img_comparison_loss(y_true, y_pred) + \
        coeff1 * flow_warped_gt_comparison_dice_loss(y_true, y_pred) + \
        coeff2 * flow_diffeomorphism_loss(y_true, y_pred)

def flow_combined_loss4(y_true, y_pred, coeff1 = 100000.0, coeff2 = 1000.0):
    return flow_warped_img_comparison_loss(y_true, y_pred) + \
        coeff2 * flow_diffeomorphism_loss(y_true, y_pred)

def infarct_classification_loss_array(y_true, y_pred, coeff = 0.99, positive_value=200.0):
    #true_max = np.amax(y_true, axis=(1,2,3))
    true_max = y_true
    true_class = np.where(np.equal(true_max, positive_value * np.ones_like(true_max)), 
                          np.ones_like(true_max), np.zeros_like(true_max))
    
    y_pred_log = np.log( coeff * np.where(np.equal(true_class, np.ones_like(true_class)), 
                                 y_pred, np.ones_like(y_pred) - y_pred) + (1.0 - coeff)/2.0 )
    return np.mean(-y_pred_log, axis=0)

def infarct_classification_loss(y_true, y_pred, coeff = 0.99, positive_value=200.0):
    #true_max = tf.reduce_max(y_true, axis=[1,2,3])
    true_max = y_true
    true_class = tf.where(K.equal(true_max, positive_value * K.ones_like(true_max)), 
                          K.ones_like(true_max), K.zeros_like(true_max))
    
    y_pred_log = tf.log( coeff * tf.where(K.equal(true_class, K.ones_like(true_class)), 
                                 y_pred, K.ones_like(y_pred) - y_pred) + (1.0 - coeff)/2.0 )
    return K.mean(-y_pred_log, axis=0)

def infarct_classification_diff_max_abs(y_true, y_pred, coeff = 0.99, positive_value=200.0):
    #true_max = tf.reduce_max(y_true, axis=[1,2,3])
    true_max = y_true
    true_class = tf.where(K.equal(true_max, positive_value * K.ones_like(true_max)), 
                          K.ones_like(true_max), K.zeros_like(true_max))
    
    diff = true_class - y_pred
    return K.max(K.abs(diff), axis=0)



def infarct_classification_loss_array2(y_true, y_pred, coeff = 0.99, positive_value=100.0):
    #true_max = np.amax(y_true, axis=(1,2,3))
    true_max = y_true
    true_class = np.where(np.greater_equal(true_max, positive_value * np.ones_like(true_max)), 
                          np.ones_like(true_max), np.zeros_like(true_max))
    
    y_pred_log = np.log( coeff * np.where(np.equal(true_class, np.ones_like(true_class)), 
                                 y_pred, np.ones_like(y_pred) - y_pred) + (1.0 - coeff)/2.0 )
    return np.mean(-y_pred_log, axis=0)

def infarct_classification_loss2(y_true, y_pred, coeff = 0.99, positive_value=100.0):
    #true_max = tf.reduce_max(y_true, axis=[1,2,3])
    true_max = y_true
    true_class = tf.where(K.greater_equal(true_max, positive_value * K.ones_like(true_max)), 
                          K.ones_like(true_max), K.zeros_like(true_max))
    
    y_pred_log = tf.log( coeff * tf.where(K.equal(true_class, K.ones_like(true_class)), 
                                 y_pred, K.ones_like(y_pred) - y_pred) + (1.0 - coeff)/2.0 )
    return K.mean(-y_pred_log, axis=0)

def infarct_classification_diff_max_abs2(y_true, y_pred, coeff = 0.99, positive_value=100.0):
    #true_max = tf.reduce_max(y_true, axis=[1,2,3])
    true_max = y_true
    true_class = tf.where(K.greater_equal(true_max, positive_value * K.ones_like(true_max)), 
                          K.ones_like(true_max), K.zeros_like(true_max))
    
    diff = true_class - y_pred
    return K.max(K.abs(diff), axis=0)




def infarct_classification_loss_array3(y_true, y_pred, coeff = 0.9999):
    y_pred_clip = coeff * y_pred + (1.0 - coeff) * 0.2
    y_pred_clip_log = np.log(y_pred_clip)

    return np.mean(np.sum(-y_true * y_pred_clip_log, axis=-1), axis = 0) 


def infarct_classification_loss3(y_true, y_pred, coeff = 0.9999):
    '''
    y_true = tf.Print(y_true, [y_true[0], y_true[1], y_true[2], tf.shape(y_true)], "Inside loss function")
    y_pred = tf.Print(y_pred, [y_pred[0], y_pred[1], y_pred[2], tf.shape(y_pred)], "Inside loss function")
    '''
    y_pred_clip = coeff * y_pred + (1.0 - coeff) * 0.2
    y_pred_clip_log = tf.log(y_pred_clip)

    return K.mean(K.sum(-y_true * y_pred_clip_log, axis=-1), axis = 0) 

def infarct_classification_diff_max_abs3(y_true, y_pred):
    '''
    y_true = tf.Print(y_true, [y_true[0], y_true[1], y_true[2], tf.shape(y_true)], "Inside metric function")
    y_pred = tf.Print(y_pred, [y_pred[0], y_pred[1], y_pred[2], tf.shape(y_pred)], "Inside metric function")
    '''
    diff = y_true - y_pred
    diff_max = K.max(K.abs(diff), axis=-1)
    return K.max(diff_max, axis=0)

def infarct_classification_diff_max_abs_array3(y_true, y_pred):
    diff = y_true - y_pred
    diff_max = np.max(np.absolute(diff), axis=-1)
    return np.max(diff_max, axis=0)



def infarct_classification_loss_array4(y_true, y_pred):
    diff = y_true - y_pred

    return np.mean(np.sum(diff * diff, axis=-1), axis = 0) 

def infarct_classification_loss4(y_true, y_pred):
    diff = y_true - y_pred

    return K.mean(K.sum(diff * diff, axis=-1), axis = 0) 

def infarct_classification_diff_max_abs4(y_true, y_pred):
    diff = y_true - y_pred
    diff_max = K.max(K.abs(diff), axis=-1)
    return K.max(diff_max, axis=0)


def euclidean_distance_loss(y_true, y_pred):
    #y_pred = (1.0 - epsilon) * y_pred + epsilon * 0.5 * K.ones_like(y_pred)
    diff = y_true - y_pred
    euclidean_dist = K.sum(diff**2, axis=[1])
    return K.mean(euclidean_dist, axis=0)

def max_distance_metric(y_true, y_pred):
    diff = y_true - y_pred
    diff_max = K.max(K.abs(diff), axis=-1)
    return K.max(diff_max, axis=0)



def flow_reconstruction_loss(y_true, y_pred):
    diff = y_true - y_pred
    euclidean_dist = K.sum(diff**2, axis=[1,2,3])
    return K.mean(euclidean_dist, axis=0)

def flow_reconstruction_loss2(y_true, y_pred):
    y_true_binary = tf.where(K.equal(y_true, K.zeros_like(y_true)), 
                             K.zeros_like(y_true), K.ones_like(y_true))
    diff = y_true - y_pred
    diff *= y_true_binary
    euclidean_dist = K.sum(diff**2, axis=[1,2,3])
    return K.mean(euclidean_dist, axis=0)

def flow_reconstruction_loss3(y_true, y_pred):
    diff = y_true - y_pred
    euclidean_dist = K.sum(diff**4, axis=[1,2,3])
    return K.mean(euclidean_dist, axis=0)


def mask_barycenter(mask, mask_value = 2.0):
    batch_size, row, column, channel = mask.shape
    barycenters = []
    for b in range(batch_size):
        #print(b)
        center_x = 0.0
        center_y = 0.0
        count = 0.0
        for r in range(row):
            for c in range(column):
                if mask[b, r, c, 0] == mask_value:
                    center_x += c
                    center_y += r
                    count += 1
        if count > 0:
            barycenters.append([center_x/count, center_y/count])
        else:
            barycenters.append([-1., -1.])
    return barycenters


def mask_barycenter2(flow, mask, mask_value = 2.0):
    batch_size, row, column, channel = flow.shape
    u = int(channel/2)
    barycenters = []
    for b in range(batch_size):
        #print(b)
        center_x = [0.0 for k in range(u)]
        center_y = [0.0 for k in range(u)]
        count = [0.0 for k in range(u)]
        for r in range(row):
            for c in range(column):
                if mask[b, r, c, 0] == mask_value:
                    for k in range(u):
                        center_x[k] += (c + flow[b, r, c, 2*k])
                        center_y[k] += (r + flow[b, r, c, 2*k+1])
                        count[k] += 1
        barycenters_b = []
        for k in range(u):
            if count[k] > 0:
                barycenters_b.append([center_x[k]/count[k], center_y[k]/count[k]])
            else:
                barycenters_b.append([-1., -1.])
        barycenters.append(barycenters_b)
    return barycenters


def masked_flow_transform(flow, mask, barycenters, lvm_value = 2.0, lvc_value = 1.0):
    batch_size, row, column, channel = flow.shape
    transformed_flow = np.zeros_like(flow)
    angles = np.ones((batch_size, row, column, 1)) * (-1.0)
    distance_flows = np.ones((batch_size, row, column, int(channel/2))) * (-1.0)
    norms = np.ones((batch_size, row, column, 1)) * (-1.0)
    boundary_pixels = np.zeros((batch_size, row, column, 1))

    row = mask.shape[1]
    column = mask.shape[2]

    for b in range(batch_size):
        center_x = barycenters[b][0]
        center_y = barycenters[b][1]
        for r in range(row):
            for c in range(column):
                if mask[b, r, c, 0] == lvm_value:
                    norm = math.sqrt((c - center_x)**2 + (r - center_y)**2)
                    cos_theta = (c - center_x) / norm
                    sin_theta = (r - center_y) / norm
                    u = int(channel/2)
                    
                    for k in range(u):
                        original_x = c + flow[b, r, c, 2*k]
                        original_y = r + flow[b, r, c, 2*k+1]
                        original_norm = math.sqrt((original_x - center_x)**2 + (original_y - center_y)**2)
                        cos_original = (original_x - center_x) / original_norm
                        sin_original = (original_y - center_y) / original_norm

                        cos_new = cos_original * cos_theta + sin_original * sin_theta
                        sin_new = sin_original * cos_theta - cos_original * sin_theta

                        transformed_flow[b, r, c, 2*k] = (original_norm / norm) * cos_new
                        transformed_flow[b, r, c, 2*k+1] = (original_norm / norm) * sin_new

                        distance_flows[b, r, c, k] = original_norm

                    if sin_theta >= 0:
                        theta = math.acos(cos_theta)
                    else:
                        theta = 2 * math.pi - math.acos(cos_theta)
                    angles[b, r, c, 0] = theta

                    norms[b, r, c, 0] = norm


                    neighbors = []
                    if r > 0:
                        neighbors.append(mask[b, r-1, c, 0])
                    if r < row - 1:
                        neighbors.append(mask[b, r+1, c, 0])
                    if c > 0:
                        neighbors.append(mask[b, r, c-1, 0])
                    if c < column - 1:
                        neighbors.append(mask[b, r, c+1, 0])

                    if lvc_value in neighbors:
                        boundary_pixels[b, r, c, 0] = -1
                    elif len(np.setdiff1d(neighbors, [lvm_value, lvc_value])) > 0:
                        boundary_pixels[b, r, c, 0] = 1
                        

    return transformed_flow, angles, distance_flows, norms, boundary_pixels



def masked_flow_transform2(flow, mask, barycenters, lvm_value = 2.0, lvc_value = 1.0):
    batch_size, row, column, channel = flow.shape
    transformed_flow = np.zeros_like(flow)
    angles = np.ones((batch_size, row, column, 1)) * (-1.0)
    distance_flows = np.ones((batch_size, row, column, int(channel/2))) * (-1.0)
    norms = np.ones((batch_size, row, column, 1)) * (-1.0)
    boundary_pixels = np.zeros((batch_size, row, column, 1))

    row = mask.shape[1]
    column = mask.shape[2]

    for b in range(batch_size):
        center_x = barycenters[b][0][0]
        center_y = barycenters[b][0][1]
        for r in range(row):
            for c in range(column):
                if mask[b, r, c, 0] == lvm_value:
                    norm = math.sqrt((c - center_x)**2 + (r - center_y)**2)
                    cos_theta = (c - center_x) / norm
                    sin_theta = (r - center_y) / norm
                    u = int(channel/2)
                    
                    for k in range(u):
                        center_x_k = barycenters[b][k][0]
                        center_y_k = barycenters[b][k][1]
                        original_x = c + flow[b, r, c, 2*k]
                        original_y = r + flow[b, r, c, 2*k+1]
                        original_norm = math.sqrt((original_x - center_x_k)**2 + (original_y - center_y_k)**2)
                        cos_original = (original_x - center_x_k) / original_norm
                        sin_original = (original_y - center_y_k) / original_norm

                        cos_new = cos_original * cos_theta + sin_original * sin_theta
                        sin_new = sin_original * cos_theta - cos_original * sin_theta

                        transformed_flow[b, r, c, 2*k] = (original_norm / norm) * cos_new
                        transformed_flow[b, r, c, 2*k+1] = (original_norm / norm) * sin_new

                        distance_flows[b, r, c, k] = original_norm

                    if sin_theta >= 0:
                        theta = math.acos(cos_theta)
                    else:
                        theta = 2 * math.pi - math.acos(cos_theta)
                    angles[b, r, c, 0] = theta

                    norms[b, r, c, 0] = norm


                    neighbors = []
                    if r > 0:
                        neighbors.append(mask[b, r-1, c, 0])
                    if r < row - 1:
                        neighbors.append(mask[b, r+1, c, 0])
                    if c > 0:
                        neighbors.append(mask[b, r, c-1, 0])
                    if c < column - 1:
                        neighbors.append(mask[b, r, c+1, 0])

                    if lvc_value in neighbors:
                        boundary_pixels[b, r, c, 0] = -1
                    elif len(np.setdiff1d(neighbors, [lvm_value, lvc_value])) > 0:
                        boundary_pixels[b, r, c, 0] = 1
                        

    return transformed_flow, angles, distance_flows, norms, boundary_pixels






def flow_by_zone(transformed_flow, angles, distance_flows, norms, boundary_pixels, num_zone, start_random = False, barycenters = None, rv_barycenters = None):
    batch_size, row, column, channel = transformed_flow.shape
    if start_random:
        start = [np.random.uniform(low = 0.0, high = 2 * math.pi) for x in range(batch_size)]
    elif (barycenters is None) or (rv_barycenters is None):
        start = [0.0 for x in range(batch_size)]
    else:
        start = []
        for b in range(batch_size):
            center_x = barycenters[b][0]
            center_y = barycenters[b][1]
            rv_center_x = rv_barycenters[b][0]
            rv_center_y = rv_barycenters[b][1]

            if rv_center_x >= 0. and rv_center_y >= 0.:
                norm = math.sqrt((rv_center_x - center_x)**2 + (rv_center_y - center_y)**2)
                cos_theta = (rv_center_x - center_x) / norm
                sin_theta = (rv_center_y - center_y) / norm
                if sin_theta >= 0:
                    theta = math.acos(cos_theta)
                else:
                    theta = 2 * math.pi - math.acos(cos_theta)
            else:
                theta = 0.
            start.append(theta)
            

    step = 2 * math.pi / num_zone
    
    zone_avg_flow = np.zeros((batch_size, num_zone, channel))
    zone_avg_inner_border_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_inner_border_normalized_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_outer_border_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_outer_border_normalized_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_myo_thickness_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_map = np.ones_like(angles) * (-1.0)
    for b in range(batch_size):
        zone_flows = [np.zeros((0, channel)) for x in range(num_zone)]
        zone_distances = [[] for x in range(num_zone)]
        zone_inner_border_flows = [np.zeros((0, int(channel/2))) for x in range(num_zone)]
        zone_inner_border_distances = [[] for x in range(num_zone)]
        zone_outer_border_flows = [np.zeros((0, int(channel/2))) for x in range(num_zone)]
        zone_outer_border_distances = [[] for x in range(num_zone)]
        for r in range(row):
            for c in range(column):
                if angles[b, r, c, 0] >= 0:
                    theta = angles[b, r, c, 0]
                    norm = norms[b, r, c, 0]
                    diff = theta - start[b]
                    if diff < 0:
                      diff += 2 * math.pi
                    zone_idx = int(math.floor(diff / step))
                    zone_flows[zone_idx] = np.concatenate((zone_flows[zone_idx], np.expand_dims(transformed_flow[b, r, c, :], axis=0)), axis=0)
                    zone_distances[zone_idx].append(norm)
                    zone_map[b, r, c, 0] = zone_idx

                    if boundary_pixels[b, r, c, 0] == -1:
                        zone_inner_border_flows[zone_idx] = np.concatenate((zone_inner_border_flows[zone_idx], np.expand_dims(distance_flows[b, r, c, :], axis=0)), axis=0)
                        zone_inner_border_distances[zone_idx].append(norm)

                    elif boundary_pixels[b, r, c, 0] == 1:
                        zone_outer_border_flows[zone_idx] = np.concatenate((zone_outer_border_flows[zone_idx], np.expand_dims(distance_flows[b, r, c, :], axis=0)), axis=0)
                        zone_outer_border_distances[zone_idx].append(norm)

        for k in range(num_zone):
            zone_avg_flow[b, k, :] = np.average(zone_flows[k], axis=0)
            zone_avg_dist = np.average(zone_distances[k])

            
            zone_avg_outer_border_flow[b, k, :] = np.average(zone_outer_border_flows[k], axis=0)
            zone_avg_outer_border_dist = np.average(zone_outer_border_distances[k])
            #zone_avg_outer_border_normalized_flow[b, k, :] = zone_avg_outer_border_flow[b, k, :] / zone_avg_dist
            zone_avg_outer_border_normalized_flow[b, k, :] = zone_avg_outer_border_flow[b, k, :] / zone_avg_outer_border_dist

            zone_avg_inner_border_flow[b, k, :] = np.average(zone_inner_border_flows[k], axis=0)
            zone_avg_inner_border_dist = np.average(zone_inner_border_distances[k])
            #zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :] / zone_avg_dist
            #zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :] / zone_avg_inner_border_dist
            zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :] / zone_avg_outer_border_dist

            
            #zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :]) / (zone_avg_outer_border_dist - zone_avg_inner_border_dist)
            zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :]) / zone_avg_outer_border_dist
            #zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :]) / zone_avg_dist


    return zone_avg_flow, zone_avg_inner_border_normalized_flow, zone_avg_outer_border_normalized_flow, zone_avg_myo_thickness_flow, zone_map




def flow_by_zone2(transformed_flow, angles, distance_flows, norms, boundary_pixels, num_zone, start_random = False, barycenters = None, rv_barycenters = None):
    batch_size, row, column, channel = transformed_flow.shape
    u = int(channel/2)
    if start_random:
        start = [np.random.uniform(low = 0.0, high = 2 * math.pi) for x in range(batch_size)]
    elif (barycenters is None) or (rv_barycenters is None):
        start = [0.0 for x in range(batch_size)]
    else:
        start = []
        for b in range(batch_size):
            center_x = barycenters[b][0][0]
            center_y = barycenters[b][0][1]
            rv_center_x = rv_barycenters[b][0][0]
            rv_center_y = rv_barycenters[b][0][1]

            if rv_center_x >= 0. and rv_center_y >= 0.:
                norm = math.sqrt((rv_center_x - center_x)**2 + (rv_center_y - center_y)**2)
                cos_theta = (rv_center_x - center_x) / norm
                sin_theta = (rv_center_y - center_y) / norm
                if sin_theta >= 0:
                    theta = math.acos(cos_theta)
                else:
                    theta = 2 * math.pi - math.acos(cos_theta)
            else:
                theta = 0.
            start.append(theta)
            

    step = 2 * math.pi / num_zone
    
    zone_avg_flow = np.zeros((batch_size, num_zone, channel))
    zone_avg_inner_border_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_inner_border_normalized_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_outer_border_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_outer_border_normalized_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_myo_thickness_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_map = np.ones_like(angles) * (-1.0)
    for b in range(batch_size):
        zone_flows = [np.zeros((0, channel)) for x in range(num_zone)]
        zone_distances = [[] for x in range(num_zone)]
        zone_inner_border_flows = [np.zeros((0, int(channel/2))) for x in range(num_zone)]
        zone_inner_border_distances = [[] for x in range(num_zone)]
        zone_outer_border_flows = [np.zeros((0, int(channel/2))) for x in range(num_zone)]
        zone_outer_border_distances = [[] for x in range(num_zone)]
        for r in range(row):
            for c in range(column):
                if angles[b, r, c, 0] >= 0:
                    theta = angles[b, r, c, 0]
                    norm = norms[b, r, c, 0]
                    diff = theta - start[b]
                    if diff < 0:
                      diff += 2 * math.pi
                    zone_idx = int(math.floor(diff / step))
                    zone_flows[zone_idx] = np.concatenate((zone_flows[zone_idx], np.expand_dims(transformed_flow[b, r, c, :], axis=0)), axis=0)
                    zone_distances[zone_idx].append(norm)
                    zone_map[b, r, c, 0] = zone_idx

                    if boundary_pixels[b, r, c, 0] == -1:
                        zone_inner_border_flows[zone_idx] = np.concatenate((zone_inner_border_flows[zone_idx], np.expand_dims(distance_flows[b, r, c, :], axis=0)), axis=0)
                        zone_inner_border_distances[zone_idx].append(norm)

                    elif boundary_pixels[b, r, c, 0] == 1:
                        zone_outer_border_flows[zone_idx] = np.concatenate((zone_outer_border_flows[zone_idx], np.expand_dims(distance_flows[b, r, c, :], axis=0)), axis=0)
                        zone_outer_border_distances[zone_idx].append(norm)

        for k in range(num_zone):
            zone_avg_flow[b, k, :] = np.average(zone_flows[k], axis=0)
            zone_avg_dist = np.average(zone_distances[k])

            
            zone_avg_outer_border_flow[b, k, :] = np.average(zone_outer_border_flows[k], axis=0)
            zone_avg_outer_border_dist = np.average(zone_outer_border_distances[k])
            #zone_avg_outer_border_normalized_flow[b, k, :] = zone_avg_outer_border_flow[b, k, :] / zone_avg_dist
            zone_avg_outer_border_normalized_flow[b, k, :] = zone_avg_outer_border_flow[b, k, :] / zone_avg_outer_border_dist

            zone_avg_inner_border_flow[b, k, :] = np.average(zone_inner_border_flows[k], axis=0)
            zone_avg_inner_border_dist = np.average(zone_inner_border_distances[k])
            #zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :] / zone_avg_dist
            #zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :] / zone_avg_inner_border_dist
            zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :] / zone_avg_outer_border_dist

            
            #zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :]) / (zone_avg_outer_border_dist - zone_avg_inner_border_dist)
            zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :]) / zone_avg_outer_border_dist
            #zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :]) / zone_avg_dist


    return zone_avg_flow, zone_avg_inner_border_normalized_flow, zone_avg_outer_border_normalized_flow, zone_avg_myo_thickness_flow, zone_map



def flow_by_zone3(transformed_flow, flow, angles, distance_flows, norms, boundary_pixels, num_zone, start_random = False, barycenters = None, rv_barycenters = None):
    batch_size, row, column, channel = transformed_flow.shape
    u = int(channel/2)
    if start_random:
        start = [np.random.uniform(low = 0.0, high = 2 * math.pi) for x in range(batch_size)]
    elif (barycenters is None) or (rv_barycenters is None):
        start = [0.0 for x in range(batch_size)]
    else:
        start = []
        for b in range(batch_size):
            center_x = barycenters[b][0][0]
            center_y = barycenters[b][0][1]
            rv_center_x = rv_barycenters[b][0][0]
            rv_center_y = rv_barycenters[b][0][1]

            if rv_center_x >= 0. and rv_center_y >= 0.:
                norm = math.sqrt((rv_center_x - center_x)**2 + (rv_center_y - center_y)**2)
                cos_theta = (rv_center_x - center_x) / norm
                sin_theta = (rv_center_y - center_y) / norm
                if sin_theta >= 0:
                    theta = math.acos(cos_theta)
                else:
                    theta = 2 * math.pi - math.acos(cos_theta)
            else:
                theta = 0.
            start.append(theta)
            

    step = 2 * math.pi / num_zone
    
    zone_avg_flow = np.zeros((batch_size, num_zone, channel))
    zone_std_original_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_inner_border_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_inner_border_normalized_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_outer_border_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_outer_border_normalized_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_avg_myo_thickness_flow = np.zeros((batch_size, num_zone, int(channel/2)))
    zone_map = np.ones_like(angles) * (-1.0)
    for b in range(batch_size):
        zone_flows = [np.zeros((0, channel)) for x in range(num_zone)]
        zone_original_flows = [np.zeros((0, channel)) for x in range(num_zone)]
        zone_distances = [[] for x in range(num_zone)]
        zone_inner_border_flows = [np.zeros((0, int(channel/2))) for x in range(num_zone)]
        zone_inner_border_distances = [[] for x in range(num_zone)]
        zone_outer_border_flows = [np.zeros((0, int(channel/2))) for x in range(num_zone)]
        zone_outer_border_distances = [[] for x in range(num_zone)]
        for r in range(row):
            for c in range(column):
                if angles[b, r, c, 0] >= 0:
                    theta = angles[b, r, c, 0]
                    norm = norms[b, r, c, 0]
                    diff = theta - start[b]
                    if diff < 0:
                      diff += 2 * math.pi
                    zone_idx = int(math.floor(diff / step))
                    zone_flows[zone_idx] = np.concatenate((zone_flows[zone_idx], np.expand_dims(transformed_flow[b, r, c, :], axis=0)), axis=0)
                    zone_original_flows[zone_idx] = np.concatenate((zone_original_flows[zone_idx], np.expand_dims(flow[b, r, c, :], axis=0)), axis=0)
                    zone_distances[zone_idx].append(norm)
                    zone_map[b, r, c, 0] = zone_idx

                    if boundary_pixels[b, r, c, 0] == -1:
                        zone_inner_border_flows[zone_idx] = np.concatenate((zone_inner_border_flows[zone_idx], np.expand_dims(distance_flows[b, r, c, :], axis=0)), axis=0)
                        zone_inner_border_distances[zone_idx].append(norm)

                    elif boundary_pixels[b, r, c, 0] == 1:
                        zone_outer_border_flows[zone_idx] = np.concatenate((zone_outer_border_flows[zone_idx], np.expand_dims(distance_flows[b, r, c, :], axis=0)), axis=0)
                        zone_outer_border_distances[zone_idx].append(norm)

        for k in range(num_zone):
            zone_avg_flow[b, k, :] = np.average(zone_flows[k], axis=0)
            zone_avg_dist = np.average(zone_distances[k])
            zone_std_original_flow[b, k, :] = np.sqrt(np.var(zone_original_flows[k][:, ::2], axis=0) + np.var(zone_original_flows[k][:, 1::2], axis=0)) / zone_avg_dist

            
            zone_avg_outer_border_flow[b, k, :] = np.average(zone_outer_border_flows[k], axis=0)
            zone_avg_outer_border_dist = np.average(zone_outer_border_distances[k])
            #zone_avg_outer_border_normalized_flow[b, k, :] = zone_avg_outer_border_flow[b, k, :] / zone_avg_dist
            #zone_avg_outer_border_normalized_flow[b, k, :] = zone_avg_outer_border_flow[b, k, :] / zone_avg_outer_border_dist
            zone_avg_outer_border_normalized_flow[b, k, :] = zone_avg_outer_border_flow[b, k, :]

            zone_avg_inner_border_flow[b, k, :] = np.average(zone_inner_border_flows[k], axis=0)
            zone_avg_inner_border_dist = np.average(zone_inner_border_distances[k])
            #zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :] / zone_avg_dist
            #zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :] / zone_avg_inner_border_dist
            #zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :] / zone_avg_outer_border_dist
            zone_avg_inner_border_normalized_flow[b, k, :] = zone_avg_inner_border_flow[b, k, :]

            
            #zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :]) / (zone_avg_outer_border_dist - zone_avg_inner_border_dist)
            #zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :]) / zone_avg_outer_border_dist
            zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :])
            #zone_avg_myo_thickness_flow[b, k, :] = (zone_avg_outer_border_flow[b, k, :] - zone_avg_inner_border_flow[b, k, :]) / zone_avg_dist


    return zone_avg_flow, zone_std_original_flow, zone_avg_inner_border_normalized_flow, zone_avg_outer_border_normalized_flow, zone_avg_myo_thickness_flow, zone_map











def avg_flow_to_arrow_map(avg_flow, avg_inner_border_normalized_flow, avg_outer_border_normalized_flow, 
        avg_myo_thickness_flow, zone_map, boundary_pixels, barycenters,
        background_files, zoned_img, output_file_prefix, zfill_num, 
        shape = (128, 128, 2),
        resolution_multiplier = 1, plot_myo_thickness = True):
    background_imgs = []
    for background_file in background_files:
        if os.path.isfile(background_file):
            #img = cv2.imread(background_file, cv2.IMREAD_GRAYSCALE)
            img = np.array(load_img2(background_file, grayscale=True, 
                                     target_size=(shape[0] * resolution_multiplier, 
                                     shape[1] * resolution_multiplier), 
                                     pad_to_square=True, resize_mode='nearest'))
        else:
            img = np.zeros((shape[0] * resolution_multiplier, 
                            shape[1] * resolution_multiplier, 1), dtype = np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        background_imgs.append(img)


    num_zone = avg_flow.shape[1]
    channel = avg_flow.shape[2]
    barycenter_x = barycenters[0][0]
    barycenter_y = barycenters[0][1]
    start_point = (int(round(barycenter_x)) * resolution_multiplier, 
                   int(round(barycenter_y)) * resolution_multiplier)
    for k in range(num_zone):
        zone_barycenter_x, zone_barycenter_y = mask_barycenter(zone_map, mask_value = k)[0]
        norm_k = math.sqrt((zone_barycenter_x - barycenter_x)**2 + 
                           (zone_barycenter_y - barycenter_y)**2)
        cos_k = (zone_barycenter_x - barycenter_x) / norm_k
        sin_k = (zone_barycenter_y - barycenter_y) / norm_k
        
        for t in range(int(channel/2)):
            if t == 0:
                t_minus = int(channel/2) - 1
            else:
                t_minus = t - 1

            previous_original_x = avg_flow[0, k, 2*t_minus]
            previous_original_y = avg_flow[0, k, 2*t_minus+1] 
            previous_inner_dist = avg_inner_border_normalized_flow[0, k, t_minus]
            previous_outer_dist = avg_outer_border_normalized_flow[0, k, t_minus]
            previous_myo_thickness = avg_myo_thickness_flow[0, k, t_minus]           

            inner_dist = avg_inner_border_normalized_flow[0, k, t]
            outer_dist = avg_outer_border_normalized_flow[0, k, t]
            myo_thickness = avg_myo_thickness_flow[0, k, t] 
            original_x = avg_flow[0, k, 2*t]
            original_y = avg_flow[0, k, 2*t+1]
            original_norm = math.sqrt(original_x**2 + original_y**2)
            original_cos_t = original_x / original_norm
            original_sin_t = original_y / original_norm

            converted_cos_t = cos_k * original_cos_t - sin_k * original_sin_t
            converted_sin_t = sin_k * original_cos_t + cos_k * original_sin_t
            converted_norm = norm_k * (original_norm / 1.0)

            converted_x = barycenter_x + converted_norm * converted_cos_t
            converted_y = barycenter_y + converted_norm * converted_sin_t
            end_point = (int(round(converted_x)) * resolution_multiplier, 
                         int(round(converted_y)) * resolution_multiplier)

            arrow_color = (0, 
                           200 + int(1000*(original_x-previous_original_x)/previous_original_x),
                           150 - int(1000*(original_x-previous_original_x)/previous_original_x))

            cv2.arrowedLine(background_imgs[t], start_point, end_point, 
                            arrow_color, thickness=1,  shift=0, tipLength=0.03)

            circle_color_inner = (200 + int(400*(inner_dist-previous_inner_dist)/previous_inner_dist),
                            50 + int(400*(inner_dist-previous_inner_dist)/previous_inner_dist),
                            50 - int(400*(inner_dist-previous_inner_dist)/previous_inner_dist))

            circle_color_outer = (200 + int(400*(outer_dist-previous_outer_dist)/previous_outer_dist),
                            50 + int(400*(outer_dist-previous_outer_dist)/previous_outer_dist),
                            50 - int(400*(outer_dist-previous_outer_dist)/previous_outer_dist))

            circle_color_myo = (200 + int(400*(myo_thickness-previous_myo_thickness)/previous_myo_thickness),
                            50 + int(400*(myo_thickness-previous_myo_thickness)/previous_myo_thickness),
                            50 - int(400*(myo_thickness-previous_myo_thickness)/previous_myo_thickness))

            if not plot_myo_thickness:
                cv2.circle(background_imgs[t], end_point, 
                           int(round(inner_dist * resolution_multiplier * 4)),
                           circle_color_inner, thickness=1, lineType=4, shift=0)
                cv2.circle(background_imgs[t], end_point, 
                           int(round(outer_dist * resolution_multiplier * 4)),
                           circle_color_outer, thickness=1, lineType=4, shift=0)
            else:
                cv2.circle(background_imgs[t], end_point, 
                           int(round(myo_thickness * resolution_multiplier * 4)),
                           circle_color_myo, thickness=1, lineType=4, shift=0)

    for t in range(int(channel/2)):
        cv2.imwrite(output_file_prefix + str(t).zfill(zfill_num) + '.png', background_imgs[t])

    
    if os.path.isfile(zoned_img):
        zoned_img_array = np.reshape(np.array(load_img2(background_file, grayscale=True, 
                                 target_size=(shape[0] * resolution_multiplier, 
                                 shape[1] * resolution_multiplier), 
                                 pad_to_square=True, resize_mode='nearest')),
                                 (shape[0] * resolution_multiplier, 
                                  shape[1] * resolution_multiplier, 1) )
    else:
        zoned_img_array = np.zeros((shape[0] * resolution_multiplier, 
                        shape[1] * resolution_multiplier, 1), dtype = np.uint8)
    zoned_img_array = np.concatenate((zoned_img_array, zoned_img_array, zoned_img_array), axis=-1)

    enlarged_zone_map = multiply_resolution(zone_map[0, :, :, 0], resolution_multiplier)
    enlarged_boundary_pixels = multiply_resolution(boundary_pixels[0, :, :, 0], resolution_multiplier)  


    #zoned_img_array[:, :, 0] += ((enlarged_zone_map + 1) * 15).astype(np.uint)
    zoned_img_array[:, :, 0] += \
        (np.where(np.equal((enlarged_zone_map + 1), np.zeros_like(enlarged_zone_map)), 
         np.zeros_like(enlarged_zone_map), (4 - (enlarged_zone_map + 1))**2) * 10).astype(np.uint)
    zoned_img_array[:, :, 1] += \
        (np.where(np.equal((enlarged_zone_map + 1), np.zeros_like(enlarged_zone_map)), 
         np.zeros_like(enlarged_zone_map), 
         np.mod((enlarged_zone_map + 1)*(enlarged_zone_map + 1), 
                7 * np.ones_like(enlarged_zone_map))) * 15).astype(np.uint)
    zoned_img_array[:, :, 2] += \
        (np.where(np.equal((enlarged_zone_map + 1), np.zeros_like(enlarged_zone_map)), 
         np.zeros_like(enlarged_zone_map), 
         np.mod((enlarged_zone_map + 1)*(enlarged_zone_map + 1), 
                5 * np.ones_like(enlarged_zone_map))) * 15).astype(np.uint)

    zoned_img_array[:, :, 0] += ((enlarged_boundary_pixels**2) * 40).astype(np.uint)
    zoned_img_array[:, :, 1] += ((enlarged_boundary_pixels**2) * 40).astype(np.uint)
    zoned_img_array[:, :, 2] += ((enlarged_boundary_pixels**2) * 40).astype(np.uint)
    cv2.imwrite(output_file_prefix + '_zones_img.png', zoned_img_array)





def avg_flow_to_arrow_map2(avg_flow, avg_inner_border_normalized_flow, avg_outer_border_normalized_flow, 
        avg_myo_thickness_flow, zone_map, boundary_pixels, barycenters,
        background_files, zoned_img, output_file_prefix, zfill_num, 
        save_zones_on_img = True, save_motion_info_on_img = True,
        shape = (128, 128, 2),
        resolution_multiplier = 1, plot_myo_thickness = True):
    background_imgs = []
    for background_file in background_files:
        if os.path.isfile(background_file):
            #img = cv2.imread(background_file, cv2.IMREAD_GRAYSCALE)
            img = np.array(load_img2(background_file, grayscale=True, 
                                     target_size=(shape[0] * resolution_multiplier, 
                                     shape[1] * resolution_multiplier), 
                                     pad_to_square=True, resize_mode='nearest'))
        else:
            img = np.zeros((shape[0] * resolution_multiplier, 
                            shape[1] * resolution_multiplier, 1), dtype = np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        background_imgs.append(img)


    num_zone = avg_flow.shape[1]
    channel = avg_flow.shape[2]

    inner_boundary_pixels = np.where( np.equal(boundary_pixels, -1.0 * np.ones_like(boundary_pixels)), 
        np.ones_like(boundary_pixels), np.zeros_like(boundary_pixels) )
    
    for k in range(num_zone):
        # Inner boundary zone
        zone_barycenter_x, zone_barycenter_y = mask_barycenter((zone_map + 1.0) * inner_boundary_pixels - 1.0, mask_value = k)[0]
        for t in range(int(channel/2)):
            barycenter_x = barycenters[0][t][0]
            barycenter_y = barycenters[0][t][1]
            start_point = (int(round(barycenter_x)) * resolution_multiplier, 
                           int(round(barycenter_y)) * resolution_multiplier)

            norm_k_t = math.sqrt((zone_barycenter_x - barycenter_x)**2 + 
                               (zone_barycenter_y - barycenter_y)**2)
            cos_k_t = (zone_barycenter_x - barycenter_x) / max(norm_k_t, 0.001)
            sin_k_t = (zone_barycenter_y - barycenter_y) / max(norm_k_t, 0.001)


            if t == 0:
                t_minus = int(channel/2) - 1
            else:
                t_minus = t - 1

            previous_inner_dist = avg_inner_border_normalized_flow[0, k, t_minus]
            previous_outer_dist = avg_outer_border_normalized_flow[0, k, t_minus]
            previous_myo_thickness = avg_myo_thickness_flow[0, k, t_minus]           

            inner_dist = avg_inner_border_normalized_flow[0, k, t]
            outer_dist = avg_outer_border_normalized_flow[0, k, t]
            myo_thickness = avg_myo_thickness_flow[0, k, t]

            converted_norm = norm_k_t * (inner_dist / avg_inner_border_normalized_flow[0, k, 0])

            converted_x = barycenter_x + converted_norm * cos_k_t
            converted_y = barycenter_y + converted_norm * sin_k_t
            end_point = (int(round(converted_x * resolution_multiplier)), 
                         int(round(converted_y * resolution_multiplier)))

            arrow_color = (0, 
                           200 + int(1000*(inner_dist-previous_inner_dist)/previous_inner_dist),
                           150 - int(1000*(inner_dist-previous_inner_dist)/previous_inner_dist))

            cv2.arrowedLine(background_imgs[t], start_point, end_point, 
                            arrow_color, thickness=2,  shift=0, tipLength=0.03)

            circle_color_inner = (200 + int(400*(inner_dist-previous_inner_dist)/previous_inner_dist),
                            50 + int(400*(inner_dist-previous_inner_dist)/previous_inner_dist),
                            50 - int(400*(inner_dist-previous_inner_dist)/previous_inner_dist))

            circle_color_outer = (200 + int(400*(outer_dist-previous_outer_dist)/previous_outer_dist),
                            50 + int(400*(outer_dist-previous_outer_dist)/previous_outer_dist),
                            50 - int(400*(outer_dist-previous_outer_dist)/previous_outer_dist))

            circle_color_myo = (200 + int(400*(myo_thickness-previous_myo_thickness)/max(previous_myo_thickness, 0.001)),
                            100 + int(600*(myo_thickness-previous_myo_thickness)/max(previous_myo_thickness, 0.001)),
                            70 - int(600*(myo_thickness-previous_myo_thickness)/max(previous_myo_thickness, 0.001)))

            if not plot_myo_thickness:
                cv2.circle(background_imgs[t], end_point, 
                           int(round(inner_dist * resolution_multiplier * 0.8)),
                           circle_color_inner, thickness=2, lineType=4, shift=0)
                cv2.circle(background_imgs[t], end_point, 
                           int(round(outer_dist * resolution_multiplier * 0.8)),
                           circle_color_outer, thickness=2, lineType=4, shift=0)
            elif myo_thickness >= 0 and previous_myo_thickness >= 0:
                cv2.circle(background_imgs[t], end_point, 
                           int(round(myo_thickness * resolution_multiplier * 0.8)),
                           circle_color_myo, thickness=2, lineType=4, shift=0)

    for t in range(int(channel/2)):
        cv2.imwrite(output_file_prefix + str(t).zfill(zfill_num) + '.png', background_imgs[t])


    if save_motion_info_on_img:
        if os.path.isfile(background_files[0]):
            #img = cv2.imread(background_file, cv2.IMREAD_GRAYSCALE)
            background_img = np.array(load_img2(background_files[0], grayscale=True, 
                                     target_size=(shape[0] * resolution_multiplier, 
                                     shape[1] * resolution_multiplier), 
                                     pad_to_square=True, resize_mode='nearest'))
        else:
            background_img = np.zeros((shape[0] * resolution_multiplier, 
                            shape[1] * resolution_multiplier, 1), dtype = np.uint8)
        background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2RGB)
        for k in range(num_zone):
            # Inner boundary zone
            zone_barycenter_x, zone_barycenter_y = mask_barycenter((zone_map + 1.0) * inner_boundary_pixels - 1.0, mask_value = k)[0]
            barycenter_x = barycenters[0][0][0]
            barycenter_y = barycenters[0][0][1]
            start_point = (int(round(barycenter_x)) * resolution_multiplier, 
                           int(round(barycenter_y)) * resolution_multiplier)

            norm_k_t = math.sqrt((zone_barycenter_x - barycenter_x)**2 + 
                               (zone_barycenter_y - barycenter_y)**2)
            cos_k_t = (zone_barycenter_x - barycenter_x) / max(norm_k_t, 0.001)
            sin_k_t = (zone_barycenter_y - barycenter_y) / max(norm_k_t, 0.001)
        
            inner_dist = avg_inner_border_normalized_flow[0, k, 0]
            inner_dist_min = avg_inner_border_normalized_flow[0, k, :].min()
            outer_dist = avg_outer_border_normalized_flow[0, k, 0]
            myo_thickness_max = avg_myo_thickness_flow[0, k, :].max()

            converted_norm = norm_k_t * (inner_dist / avg_inner_border_normalized_flow[0, k, 0])
            converted_x = barycenter_x + converted_norm * cos_k_t
            converted_y = barycenter_y + converted_norm * sin_k_t
            end_point = (int(round(converted_x * resolution_multiplier)), 
                         int(round(converted_y * resolution_multiplier)))

            converted_norm_min = norm_k_t * (inner_dist_min / avg_inner_border_normalized_flow[0, k, 0])
            converted_x_min = barycenter_x + converted_norm_min * cos_k_t
            converted_y_min = barycenter_y + converted_norm_min * sin_k_t
            end_point_min = (int(round(converted_x_min * resolution_multiplier)), 
                             int(round(converted_y_min * resolution_multiplier)))

            arrow_color = (0, 200, 100)
            arrow_color_min = (0, 100, 200)
            circle_color_myo = (250, 150, 20)

            cv2.arrowedLine(background_img, start_point, end_point, 
                            arrow_color, thickness=2,  shift=0, tipLength=0.2)

            cv2.arrowedLine(background_img, start_point, end_point_min, 
                            arrow_color_min, thickness=3,  shift=0, tipLength=0.13, line_type=8)

            cv2.circle(background_img, end_point, 
                       int(round(myo_thickness_max * resolution_multiplier * 0.8)),
                       circle_color_myo, thickness=2, lineType=4, shift=0)

            cv2.imwrite(output_file_prefix + 'motion_info.png', background_img)


    if save_zones_on_img:
        if os.path.isfile(zoned_img):
            zoned_img_array = np.reshape(np.array(load_img2(background_file, grayscale=True, 
                                     target_size=(shape[0] * resolution_multiplier, 
                                     shape[1] * resolution_multiplier), 
                                     pad_to_square=True, resize_mode='nearest')),
                                     (shape[0] * resolution_multiplier, 
                                      shape[1] * resolution_multiplier, 1) )
        else:
            zoned_img_array = np.zeros((shape[0] * resolution_multiplier, 
                            shape[1] * resolution_multiplier, 1), dtype = np.uint8)
        zoned_img_array = np.concatenate((zoned_img_array, zoned_img_array, zoned_img_array), axis=-1)

        enlarged_zone_map = multiply_resolution(zone_map[0, :, :, 0], resolution_multiplier)
        enlarged_boundary_pixels = multiply_resolution(boundary_pixels[0, :, :, 0], resolution_multiplier)  


        '''
        zoned_img_array[:, :, 0] += \
            (np.where(np.equal((enlarged_zone_map + 1), np.zeros_like(enlarged_zone_map)), 
             np.zeros_like(enlarged_zone_map), (4 - (enlarged_zone_map + 1))**2) * 10).astype(np.uint)
        zoned_img_array[:, :, 1] += \
            (np.where(np.equal((enlarged_zone_map + 1), np.zeros_like(enlarged_zone_map)), 
             np.zeros_like(enlarged_zone_map), 
             np.mod((enlarged_zone_map + 1)*(enlarged_zone_map + 1), 
                    7 * np.ones_like(enlarged_zone_map))) * 15).astype(np.uint)
        zoned_img_array[:, :, 2] += \
            (np.where(np.equal((enlarged_zone_map + 1), np.zeros_like(enlarged_zone_map)), 
             np.zeros_like(enlarged_zone_map), 
             np.mod((enlarged_zone_map + 1)*(enlarged_zone_map + 1), 
                    5 * np.ones_like(enlarged_zone_map))) * 15).astype(np.uint)
        '''
        
        #bgr_color_list = \
        #    [(80, 0, 0), (0, 80, 80), (0, 0, 80), (60, 60, 0), (60, 0, 60), (0, 80, 0)]
        # [blue, yellow, red, cyan, violet, green]
        bgr_color_list = \
            [(60, 10, 0), (0, 40, 40), (0, 0, 40), (40, 30, 0), (30, 0, 30), (0, 40, 0)]
        zoned_img_array = add_color_to_bgr_channels(zoned_img_array, enlarged_zone_map, range(6), bgr_color_list)

        zoned_img_array[:, :, 0] += ((enlarged_boundary_pixels**2) * 40).astype(np.uint)
        zoned_img_array[:, :, 1] += ((enlarged_boundary_pixels**2) * 40).astype(np.uint)
        zoned_img_array[:, :, 2] += ((enlarged_boundary_pixels**2) * 40).astype(np.uint)
        zoned_img_array = np.where(zoned_img_array > np.ones_like(zoned_img_array) * 255, np.ones_like(zoned_img_array) * 255, zoned_img_array)
        cv2.imwrite(output_file_prefix + 'zones.png', zoned_img_array)












def avg_flow_to_arrow_map3(avg_flow, avg_inner_border_normalized_flow, avg_outer_border_normalized_flow, 
        avg_myo_thickness_flow, zone_map, boundary_pixels, barycenters,
        background_files, zoned_img, output_file_prefix, zfill_num, 
        es_instant_in_range,
        save_zones_on_img = True, save_motion_info_on_img = True,
        shape = (128, 128, 2),
        resolution_multiplier = 1, plot_myo_thickness = True):
    background_imgs = []
    for background_file in background_files:
        if os.path.isfile(background_file):
            #img = cv2.imread(background_file, cv2.IMREAD_GRAYSCALE)
            img = np.array(load_img2(background_file, grayscale=True, 
                                     target_size=(shape[0] * resolution_multiplier, 
                                     shape[1] * resolution_multiplier), 
                                     pad_to_square=True, resize_mode='nearest'))
        else:
            img = np.zeros((shape[0] * resolution_multiplier, 
                            shape[1] * resolution_multiplier, 1), dtype = np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        background_imgs.append(img)


    num_zone = avg_flow.shape[1]
    channel = avg_flow.shape[2]

    inner_boundary_pixels = np.where( np.equal(boundary_pixels, -1.0 * np.ones_like(boundary_pixels)), 
        np.ones_like(boundary_pixels), np.zeros_like(boundary_pixels) )
    

    bgr_color_list1 = \
            [(255, 40, 40), (0, 240, 240), (0, 0, 255), (240, 240, 0), (240, 0, 240), (0, 200, 0)]
    for k in range(num_zone):
        # Inner boundary zone
        zone_barycenter_x, zone_barycenter_y = mask_barycenter((zone_map + 1.0) * inner_boundary_pixels - 1.0, mask_value = k)[0]
        for t in range(int(channel/2)):
            barycenter_x = barycenters[0][t][0]
            barycenter_y = barycenters[0][t][1]
            start_point = (int(round(barycenter_x)) * resolution_multiplier, 
                           int(round(barycenter_y)) * resolution_multiplier)

            norm_k_t = math.sqrt((zone_barycenter_x - barycenter_x)**2 + 
                               (zone_barycenter_y - barycenter_y)**2)
            cos_k_t = (zone_barycenter_x - barycenter_x) / max(norm_k_t, 0.001)
            sin_k_t = (zone_barycenter_y - barycenter_y) / max(norm_k_t, 0.001)


            if t == 0:
                t_minus = int(channel/2) - 1
            else:
                t_minus = t - 1

            previous_inner_dist = avg_inner_border_normalized_flow[0, k, t_minus]
            previous_outer_dist = avg_outer_border_normalized_flow[0, k, t_minus]
            previous_myo_thickness = avg_myo_thickness_flow[0, k, t_minus] 
            ed_myo_thickness = avg_myo_thickness_flow[0, k, 0]          

            inner_dist = avg_inner_border_normalized_flow[0, k, t]
            outer_dist = avg_outer_border_normalized_flow[0, k, t]
            myo_thickness = avg_myo_thickness_flow[0, k, t]

            converted_norm = norm_k_t * (inner_dist / avg_inner_border_normalized_flow[0, k, 0])

            converted_x = barycenter_x + converted_norm * cos_k_t
            converted_y = barycenter_y + converted_norm * sin_k_t
            end_point = (int(round(converted_x * resolution_multiplier)), 
                         int(round(converted_y * resolution_multiplier)))

            
            cv2.arrowedLine(background_imgs[t], start_point, end_point, 
                            bgr_color_list1[k], thickness=2,  shift=0, tipLength=0.03)

            circle_color_inner = (200 + int(400*(inner_dist-previous_inner_dist)/previous_inner_dist),
                            50 + int(400*(inner_dist-previous_inner_dist)/previous_inner_dist),
                            50 - int(400*(inner_dist-previous_inner_dist)/previous_inner_dist))

            circle_color_outer = (200 + int(400*(outer_dist-previous_outer_dist)/previous_outer_dist),
                            50 + int(400*(outer_dist-previous_outer_dist)/previous_outer_dist),
                            50 - int(400*(outer_dist-previous_outer_dist)/previous_outer_dist))

            #circle_color_myo = (255,204,153)
            circle_color_myo = bgr_color_list1[k]

            if not plot_myo_thickness:
                cv2.circle(background_imgs[t], end_point, 
                           int(round(inner_dist * resolution_multiplier * 0.8)),
                           circle_color_inner, thickness=2, lineType=4, shift=0)
                cv2.circle(background_imgs[t], end_point, 
                           int(round(outer_dist * resolution_multiplier * 0.8)),
                           circle_color_outer, thickness=2, lineType=4, shift=0)
            elif myo_thickness >= 0 and ed_myo_thickness >= 0:
                cv2.circle(background_imgs[t], end_point, 
                           int(round(max(myo_thickness-ed_myo_thickness, 0) * resolution_multiplier * 0.8)),
                           circle_color_myo, thickness=2, lineType=4, shift=0)

    for t in range(int(channel/2)):
        cv2.imwrite(output_file_prefix + str(t).zfill(zfill_num) + '.png', background_imgs[t])


    if save_motion_info_on_img:
        if os.path.isfile(background_files[0]):
            #img = cv2.imread(background_file, cv2.IMREAD_GRAYSCALE)
            background_img = np.array(load_img2(background_files[0], grayscale=True, 
                                     target_size=(shape[0] * resolution_multiplier, 
                                     shape[1] * resolution_multiplier), 
                                     pad_to_square=True, resize_mode='nearest'))
        else:
            background_img = np.zeros((shape[0] * resolution_multiplier, 
                            shape[1] * resolution_multiplier, 1), dtype = np.uint8)
        background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2RGB)
        for k in range(num_zone):
            # Inner boundary zone
            zone_barycenter_x, zone_barycenter_y = mask_barycenter((zone_map + 1.0) * inner_boundary_pixels - 1.0, mask_value = k)[0]
            barycenter_x = barycenters[0][0][0]
            barycenter_y = barycenters[0][0][1]
            start_point = (int(round(barycenter_x)) * resolution_multiplier, 
                           int(round(barycenter_y)) * resolution_multiplier)

            norm_k_t = math.sqrt((zone_barycenter_x - barycenter_x)**2 + 
                               (zone_barycenter_y - barycenter_y)**2)
            cos_k_t = (zone_barycenter_x - barycenter_x) / max(norm_k_t, 0.001)
            sin_k_t = (zone_barycenter_y - barycenter_y) / max(norm_k_t, 0.001)
        
            inner_dist = avg_inner_border_normalized_flow[0, k, 0]
            inner_dist_min = avg_inner_border_normalized_flow[0, k, :].min()
            outer_dist = avg_outer_border_normalized_flow[0, k, 0]
            ed_myo_thickness = avg_myo_thickness_flow[0, k, 0]
            es_myo_thickness = avg_myo_thickness_flow[0, k, es_instant_in_range]

            converted_norm = norm_k_t * (inner_dist / avg_inner_border_normalized_flow[0, k, 0])
            converted_x = barycenter_x + converted_norm * cos_k_t
            converted_y = barycenter_y + converted_norm * sin_k_t
            end_point = (int(round(converted_x * resolution_multiplier)), 
                         int(round(converted_y * resolution_multiplier)))

            converted_norm_min = norm_k_t * (inner_dist_min / avg_inner_border_normalized_flow[0, k, 0])
            converted_x_min = barycenter_x + converted_norm_min * cos_k_t
            converted_y_min = barycenter_y + converted_norm_min * sin_k_t
            end_point_min = (int(round(converted_x_min * resolution_multiplier)), 
                             int(round(converted_y_min * resolution_multiplier)))

            arrow_color = (153, 255, 153)
            arrow_color_min = (102, 178, 255)
            #circle_color_myo = (255,204,153)
            circle_color_myo = bgr_color_list1[k]

            cv2.arrowedLine(background_img, start_point, end_point, 
                            arrow_color, thickness=2,  shift=0, tipLength=0.25)
            cv2.arrowedLine(background_img, start_point, end_point, 
                            bgr_color_list1[k], thickness=2,  shift=0, tipLength=0.0)

            cv2.arrowedLine(background_img, start_point, end_point_min, 
                            arrow_color_min, thickness=2,  shift=0, tipLength=0.22)
            cv2.arrowedLine(background_img, start_point, end_point_min, 
                            bgr_color_list1[k], thickness=2,  shift=0, tipLength=0.0)

            cv2.circle(background_img, end_point, 
                       int(round(max(es_myo_thickness-ed_myo_thickness,0) * resolution_multiplier * 0.8)),
                       circle_color_myo, thickness=2, lineType=4, shift=0)

            cv2.imwrite(output_file_prefix + 'motion_info.png', background_img)


    if save_zones_on_img:
        if os.path.isfile(zoned_img):
            zoned_img_array = np.reshape(np.array(load_img2(background_files[0], grayscale=True, 
                                     target_size=(shape[0] * resolution_multiplier, 
                                     shape[1] * resolution_multiplier), 
                                     pad_to_square=True, resize_mode='nearest')),
                                     (shape[0] * resolution_multiplier, 
                                      shape[1] * resolution_multiplier, 1) )
        else:
            zoned_img_array = np.zeros((shape[0] * resolution_multiplier, 
                            shape[1] * resolution_multiplier, 1), dtype = np.uint8)
        zoned_img_array = np.concatenate((zoned_img_array, zoned_img_array, zoned_img_array), axis=-1)

        enlarged_zone_map = multiply_resolution(zone_map[0, :, :, 0], resolution_multiplier)
        enlarged_boundary_pixels = multiply_resolution(boundary_pixels[0, :, :, 0], resolution_multiplier)  


        '''
        zoned_img_array[:, :, 0] += \
            (np.where(np.equal((enlarged_zone_map + 1), np.zeros_like(enlarged_zone_map)), 
             np.zeros_like(enlarged_zone_map), (4 - (enlarged_zone_map + 1))**2) * 10).astype(np.uint)
        zoned_img_array[:, :, 1] += \
            (np.where(np.equal((enlarged_zone_map + 1), np.zeros_like(enlarged_zone_map)), 
             np.zeros_like(enlarged_zone_map), 
             np.mod((enlarged_zone_map + 1)*(enlarged_zone_map + 1), 
                    7 * np.ones_like(enlarged_zone_map))) * 15).astype(np.uint)
        zoned_img_array[:, :, 2] += \
            (np.where(np.equal((enlarged_zone_map + 1), np.zeros_like(enlarged_zone_map)), 
             np.zeros_like(enlarged_zone_map), 
             np.mod((enlarged_zone_map + 1)*(enlarged_zone_map + 1), 
                    5 * np.ones_like(enlarged_zone_map))) * 15).astype(np.uint)
        '''
        
        #bgr_color_list2 = \
        #    [(80, 0, 0), (0, 80, 80), (0, 0, 80), (60, 60, 0), (60, 0, 60), (0, 80, 0)]
        # [blue, yellow, red, cyan, violet, green]
        bgr_color_list2 = \
            [(60, 10, 0), (0, 40, 40), (0, 0, 40), (40, 30, 0), (30, 0, 30), (0, 40, 0)]
        zoned_img_array = add_color_to_bgr_channels(zoned_img_array, enlarged_zone_map, range(6), bgr_color_list2)

        zoned_img_array[:, :, 0] += ((enlarged_boundary_pixels**2) * 40).astype(np.uint)
        zoned_img_array[:, :, 1] += ((enlarged_boundary_pixels**2) * 40).astype(np.uint)
        zoned_img_array[:, :, 2] += ((enlarged_boundary_pixels**2) * 40).astype(np.uint)
        zoned_img_array = np.where(zoned_img_array > np.ones_like(zoned_img_array) * 255, np.ones_like(zoned_img_array) * 255, zoned_img_array)
        cv2.imwrite(output_file_prefix + 'zones.png', zoned_img_array)









def add_color_to_bgr_channels(array, zone_map, zone_idx_list, bgr_color_list):
    for k in range(len(zone_idx_list)):
        idx = zone_idx_list[k]
        bgr_color = bgr_color_list[k]
        for p in range(3):
            array[:, :, p] += \
                (np.where(np.equal(zone_map, idx * np.ones_like(zone_map)), 
                 bgr_color[p] * np.ones_like(zone_map), np.zeros_like(zone_map))).astype(np.uint)
    return array
    

def multiply_resolution(img_array, resolution_multiplier=1):
    row, column = img_array.shape
    output = np.zeros((row * resolution_multiplier, column * resolution_multiplier))
    for r in range(row * resolution_multiplier):
        for c in range(column * resolution_multiplier):
            output[r, c] = img_array[int(np.floor(r/resolution_multiplier)),
                                     int(np.floor(c/resolution_multiplier))]
    return output


def flow_to_arrow_map(flow_file, background_file, mask_file, enlarge_mask, enlarge_width,
        output_file, step = 5, shape = (128, 128, 2), resolution_multiplier = 1):
    if os.path.isfile(flow_file):
        flow = np.load(flow_file)
        flow = np.reshape(flow, shape)
    else:
        flow = np.zeros(shape)

    if os.path.isfile(background_file):
        #img = cv2.imread(background_file, cv2.IMREAD_GRAYSCALE)
        img = np.array(load_img2(background_file, grayscale=True, 
                                 target_size=(shape[0] * resolution_multiplier, 
                                 shape[1] * resolution_multiplier), 
                                 pad_to_square=True, resize_mode='nearest'))
    else:
        img = np.zeros((shape[0] * resolution_multiplier, 
                        shape[1] * resolution_multiplier, 1), dtype = np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if os.path.isfile(mask_file):
        mask = load_img2(mask_file, grayscale=True, 
                         target_size=(shape[0] * resolution_multiplier, 
                         shape[1] * resolution_multiplier),
                         pad_to_square=True, resize_mode='nearest')
        mask = np.array(mask)
        mask = np.where(np.equal(mask, 100.0 * np.ones_like(mask)), 
                        np.ones_like(mask), np.zeros_like(mask))

        # Enlarge the LVM mask if necessary
        if enlarge_mask:
            mask = np.reshape(mask, (1, shape[0] * resolution_multiplier, 
                                     shape[1] * resolution_multiplier, 1))
            mask = enlarge_mask3(mask, width=enlarge_width * resolution_multiplier,
                                 enlarge_value=1.0)
            mask = np.reshape(mask, (shape[0] * resolution_multiplier, 
                                     shape[1] * resolution_multiplier))

        for r in range(0, shape[0], step):
            for c in range(0, shape[1], step):
                if mask[r * resolution_multiplier, c * resolution_multiplier] == 1:
                    start_point = (c * resolution_multiplier, r * resolution_multiplier)
                    end_point = (int(round((c + flow[r, c, 0]) * resolution_multiplier)), 
                                 int(round((r + flow[r, c, 1]) * resolution_multiplier)))
                    cv2.arrowedLine(img, start_point, end_point, 
                                    (0,0,255), thickness=2,  shift=0, tipLength=0.3)

    else:
        for r in range(0, shape[0], step):
            for c in range(0, shape[1], step):
                start_point = (c * resolution_multiplier, r * resolution_multiplier)
                end_point = (int(round((c + flow[r, c, 0]) * resolution_multiplier)), 
                             int(round((r + flow[r, c, 1]) * resolution_multiplier)))
                cv2.arrowedLine(img, start_point, end_point, 
                                (0,0,255), thickness=2,  shift=0, tipLength=0.3)


    cv2.imwrite(output_file, img)


def img_to_video(img_list, output_file, is_grayscale = False):
    if is_grayscale:
        read = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
        height, width = read.shape
    else:
        read = cv2.imread(img_list[0], cv2.IMREAD_COLOR)
        height, width, channels = read.shape
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #output = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))
    output = cv2.VideoWriter(output_file, -1, 1, (width, height))
    for img_path in img_list:
        img = cv2.imread(img_path)
        output.write(img)
    output.release()
    return output

def img_to_video2(img_name_format, output_file):
    os.system("ffmpeg -r 1 -i " + img_name_format + " -vcodec mpeg4 -y " + output_file)

def img_to_video3(img_list, output_file):
    imgs = []
    for img_path in img_list:
        imgs.append(imageio.imread(img_path))
    imageio.mimsave(output_file, imgs, format='GIF', duration=0.15)




def enlarge_mask(mask, width):
    batch_size = mask.shape[0]
    row = mask.shape[1]
    column = mask.shape[2]

    note = np.copy(mask)
    output = np.copy(mask)
    for i in range(width):
        for k in range(batch_size):            
            for r in range(row):
                for c in range(column):
                    if note[k, r, c, 0] == 1:
                        if r-1 >= 0 and note[k, r-1, c, 0] == 0:
                            output[k, r-1, c, 0] = 1
                        if r+1 < row and note[k, r+1, c, 0] == 0:
                            output[k, r+1, c, 0] = 1
                        if c-1 >= 0 and note[k, r, c-1, 0] == 0:
                            output[k, r, c-1, 0] = 1
                        if c+1 < column and note[k, r, c+1, 0] == 0:
                            output[k, r, c+1, 0] = 1
        note = np.copy(output)
    return output


def enlarge_mask2(mask, width):
    batch_size = mask.shape[0]
    row = mask.shape[1]
    column = mask.shape[2]

    output = np.zeros_like(mask)
    for k in range(batch_size):            
        for r in range(row):
            for c in range(column):
                neighborhood = mask[k, max(r - width, 0):min(r + width + 1, row),
                                    max(c - width, 0):min(c + width + 1, column), 0]
                output[k, r, c, 0] = np.amax(neighborhood)
    return output


def enlarge_mask3(mask, width, enlarge_value):
    batch_size = mask.shape[0]
    row = mask.shape[1]
    column = mask.shape[2]

    output = np.copy(mask)
    for k in range(batch_size):            
        for r in range(row):
            for c in range(column):
                if mask[k, r, c, 0] == enlarge_value:
                    for rr in range(max(r - width, 0), min(r + width + 1, row)):
                        for cc in range(max(c - width, 0), min(c + width + 1, column)):
                            output[k, rr, cc, 0] = enlarge_value
    return output

def enlarge_mask4(mask, width, enlarge_value, neighbor_values):
    batch_size = mask.shape[0]
    row = mask.shape[1]
    column = mask.shape[2]

    output = np.copy(mask)
    for k in range(batch_size):            
        for r in range(row):
            for c in range(column):
                if mask[k, r, c, 0] == enlarge_value:
                    for rr in range(max(r - width, 0), min(r + width + 1, row)):
                        for cc in range(max(c - width, 0), min(c + width + 1, column)):
                            if mask[k, rr, cc, 0] in neighbor_values:
                                output[k, rr, cc, 0] = enlarge_value
    return output


def myo_mask_max_min_mean_thickness(mask, myo_value = 2, lvc_value = 1 ,bg_value = 0, rv_value = 3):
    h, w = mask.shape
    myo_lvc_boundary = []
    myo_bg_boundary = []
    for hh in range(h):
        for ww in range(w):
            if (mask[hh, ww] == lvc_value) and \
               ((hh > 0 and mask[hh-1, ww] == myo_value) or \
                (hh < (h-1) and mask[hh+1, ww] == myo_value) or \
                (ww > 0 and mask[hh, ww-1] == myo_value) or \
                (ww < (ww-1) and mask[hh, ww+1] == myo_value)):
                myo_lvc_boundary.append((hh, ww))

            if (mask[hh, ww] == myo_value) and \
               ((hh > 0 and mask[hh-1, ww] in [bg_value, rv_value]) or \
                (hh < (h-1) and mask[hh+1, ww] in [bg_value, rv_value]) or \
                (ww > 0 and mask[hh, ww-1] in [bg_value, rv_value]) or \
                (ww < (ww-1) and mask[hh, ww+1] in [bg_value, rv_value])):
                myo_bg_boundary.append((hh, ww))

    max_thickness = -1.0
    min_thickness = -1.0
    sum_thickness = 0.0
    for u in myo_lvc_boundary:
        min_dist = float(max(h, w))
        for v in myo_bg_boundary:
            dist = math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
            min_dist = min(min_dist, dist)
        max_thickness = max(max_thickness, min_dist)
        if min_thickness < 0:
            min_thickness = min_dist
        else:
            min_thickness = min(min_thickness, min_dist)
        sum_thickness += min_dist

    if len(myo_lvc_boundary) > 0:
        mean_thickness = sum_thickness / len(myo_lvc_boundary)
    else:
        mean_thickness = -1.0
    return max_thickness, min_thickness, mean_thickness


def volume_calculation_given_slice_area(area_list, thickness):
    length = len(area_list)
    volume = 0.0
    for l in range(0, length-1):
        subvolume = (area_list[l] + area_list[l+1] + math.sqrt(area_list[l] * area_list[l+1])) * thickness / 3
        volume += subvolume
    return volume



def number_of_components(array, value):
    v_mask = np.where(array==value, np.ones_like(array), np.zeros_like(array))
    connected_components, num_connected_components = ndimage.label(v_mask)
    return num_connected_components

def keep_largest_components(array, keep_values=[1, 2, 3], values=[1, 2, 3]):
    output = np.zeros_like(array)
    for v in values:
        v_mask = np.where(array==v, np.ones_like(array), np.zeros_like(array))
        connected_components, num_connected_components = ndimage.label(v_mask)
        if (num_connected_components > 1) and (v in keep_values):
            unique, counts = np.unique(connected_components, return_counts=True)
            max_idx = np.where(counts == max(counts[1:]))[0][0]
            v_mask = v_mask * (connected_components == max_idx)
        output = output + v * v_mask
    return output


def second_largest_component_ratio(array, value):
    v_mask = np.where(array==value, np.ones_like(array), np.zeros_like(array))
    connected_components, num_connected_components = ndimage.label(v_mask)
    if (num_connected_components > 1):
        unique, counts = np.unique(connected_components, return_counts=True)
        max_idx = np.where(counts == max(counts[1:]))[0][0]
        max_count = (connected_components == max_idx).sum()
        second_tmp_idx = counts[1:].argsort()[-2]
        second_max_idx = np.where(counts == counts[1:][second_tmp_idx])[0][0]
        second_max_count = (connected_components == second_max_idx).sum()
        return float(second_max_count) / max_count
    else:
        return 0.0

def second_largest_component_count(array, value):
    v_mask = np.where(array==value, np.ones_like(array), np.zeros_like(array))
    connected_components, num_connected_components = ndimage.label(v_mask)
    if (num_connected_components > 1):
        unique, counts = np.unique(connected_components, return_counts=True)
        max_idx = np.where(counts == max(counts[1:]))[0][0]
        max_count = (connected_components == max_idx).sum()
        second_tmp_idx = counts[1:].argsort()[-2]
        second_max_idx = np.where(counts == counts[1:][second_tmp_idx])[0][0]
        second_max_count = (connected_components == second_max_idx).sum()
        return second_max_count
    else:
        return 0.0


def v1_touch_v2(array, size_x, size_y, v1, v2, threshold=10):
    touch_count = 0
    for p in range(size_x * size_y):
        r = p // size_x
        c = p % size_x
        if array[r, c] == v1:
            up_touch = (r != 0) and (array[r-1, c] == v2)
            down_touch = (r != size_y-1) and (array[r+1, c] == v2)
            left_touch = (c != 0) and (array[r, c-1] == v2)
            right_touch = (c != size_x-1) and (array[r, c+1] == v2)

            touch_count += (up_touch + down_touch + left_touch + right_touch)
    return touch_count >= threshold


def touch_length_count(array, size_x, size_y, v1, v2):
    touch_count = 0
    for p in range(size_x * size_y):
        r = p // size_x
        c = p % size_x
        if array[r, c] == v1:
            up_touch = (r != 0) and (array[r-1, c] == v2)
            down_touch = (r != size_y-1) and (array[r+1, c] == v2)
            left_touch = (c != 0) and (array[r, c-1] == v2)
            right_touch = (c != size_x-1) and (array[r, c+1] == v2)

            touch_count += (up_touch + down_touch + left_touch + right_touch)
    return touch_count

def area_boundary_ratio(array, size_x, size_y, v):
    area = 0
    boundary = 0
    for p in range(size_x * size_y):
        r = p // size_x
        c = p % size_x
        if array[r, c] == v:
            up_touch = (r != 0) and (array[r-1, c] != v)
            down_touch = (r != size_y-1) and (array[r+1, c] != v)
            left_touch = (c != 0) and (array[r, c-1] != v)
            right_touch = (c != size_x-1) and (array[r, c+1] != v)

            area += 1
            boundary += (up_touch + down_touch + left_touch + right_touch)
    return float(area)/(boundary*boundary)

def change_neighbor_value(array, size_x, size_y, v0, v1, v2):
    output = array
    for p in range(size_x * size_y):
        r = p // size_x
        c = p % size_x
        if array[r, c] == v0:
            if (r != 0) and (array[r-1, c] == v1):
                output[r-1, c] = v2
            if (r != size_y-1) and (array[r+1, c] == v1):
                output[r+1, c] = v2
            if (c != 0) and (array[r, c-1] == v1):
                output[r, c-1] = v2
            if (c != size_x-1) and (array[r, c+1] == v1):
                output[r, c+1] = v2
    return output

def pixel_count_by_value(array, value):
    count = 0
    array = array.flatten()
    for v in array:
        if v == value:
            count += 1
    return count


def save_layer_output(model, data, layer_name='output', save_path_prefix='output'):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data)
    batch_size = intermediate_output.shape[0]
    for i in range(batch_size):
        data = intermediate_output[i]
        if len(data.shape) == 3:
            if (data.shape[2] == 1):
                img = array_to_img(data, data_format=None, scale=True)
                img.save(save_path_prefix + str(i).zfill(2) + ".png")
                np.savetxt(save_path_prefix + str(i).zfill(2) + ".txt", data, fmt='%.6f')
            else:
                for j in range(data.shape[2]):
                    data_j = data[:,:,j:(j+1)]
                    img_j = array_to_img(data_j, data_format=None, scale=True)
                    img_j.save(save_path_prefix + str(i).zfill(2) + "_" + str(j).zfill(2) + ".png")
                    np.savetxt(save_path_prefix + str(i).zfill(2) + "_" + str(j).zfill(2) + ".txt", data_j, fmt='%.6f')
        else:
            np.savetxt(save_path_prefix + str(i).zfill(2) + ".txt", data, fmt='%.6f')


def print_layer_output(model, data, layer_name='output'):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data)
    batch_size = intermediate_output.shape[0]
    for i in range(batch_size):
        data = intermediate_output[i]
        print(data)


def print_model_weights(model):
    layers = model.layers[-10:]
    all_weights = []
    for layer in layers:
        all_weights += layer.weights
    evaluated_weights = K.batch_get_value(all_weights)
    for layer in layers:
        print("\nlayer #{} {}:{}".format(model.layers.index(layer), layer.__class__.__name__, layer.name) )
        print("input shape: {}".format(layer.input_shape) )
        print("output shape: {}".format(layer.output_shape) )
        for j, w in enumerate(layer.trainable_weights):
            w = evaluated_weights[all_weights.index(w)]
            wf = w.flatten()
            s = w.size
            print(" weights #{} {}: {} {} {} {}".format(j, w.shape,\
                wf[int(s//5)], wf[int(2*s//5)], wf[int(3*s//5)], wf[int(4*s//5)]) )


def print_model_gradients(model, batch_img, batch_mask):
    layers = model.layers[-10:]
    all_trainable_weights = []
    for layer in layers:
        all_trainable_weights += layer.trainable_weights
    gradient_list = model.optimizer.get_gradients(model.total_loss,
                                                  all_trainable_weights)
    '''
    get_gradient_list = K.function(
        inputs=[model.inputs[0], model.targets[0], model.sample_weights[0],
                K.learning_phase()],
        outputs=gradient_list,
        updates=None)
    evaluated_gradient_list = get_gradient_list(
        [batch_img, batch_mask, [1.0]*(batch_img.shape[0]), 1])
    '''
    evaluated_gradient_list = K.get_session().run(
         gradient_list, 
         feed_dict={model.inputs[0]: batch_img,
                    model.targets[0]: batch_mask,
                    model.sample_weights[0]: [1.0]*(batch_img.shape[0]),
                    K.learning_phase(): 1})

    for layer in layers:
        print("\nlayer #{} {}:{}".format(model.layers.index(layer), layer.__class__.__name__, layer.name) )
        print("input shape: {}".format(layer.input_shape) )
        print("output shape: {}".format(layer.output_shape) )

        for j, w in enumerate(layer.trainable_weights):
            g = evaluated_gradient_list[all_trainable_weights.index(w)]
            gf = g.flatten()
            s = g.size
            print(" gradients #{} {}: {} {} {} {}".format(j, g.shape,\
                gf[int(s//5)], gf[int(2*s//5)], gf[int(3*s//5)], gf[int(4*s//5)]) )




def print_model_weights_gradients(model, batch_img, batch_mask):
    layers = model.layers[-10:]

    all_weights = []
    all_trainable_weights = []
    for layer in layers:
        all_weights += layer.weights
        all_trainable_weights += layer.trainable_weights
    gradient_list = model.optimizer.get_gradients(model.total_loss,
                                                  all_trainable_weights)
    weights_len = len(all_weights)
    gradient_len = len(gradient_list)
    if (weights_len + gradient_len > 0):
        if not isinstance(batch_img, list):
            evaluated = K.get_session().run(
                all_weights + gradient_list, 
                feed_dict={model.inputs[0]: batch_img,
                           model.targets[0]: batch_mask,
                           model.sample_weights[0]: [1.0]*(batch_img.shape[0]),
                           K.learning_phase(): 1})

        elif not isinstance(batch_mask, list):
            if len(batch_img) == 2:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.targets[0]: batch_mask,
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 3:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.targets[0]: batch_mask,
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 4:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.inputs[3]: batch_img[3],
                               model.targets[0]: batch_mask,
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 9:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.inputs[3]: batch_img[3],
                               model.inputs[4]: batch_img[4],
                               model.inputs[5]: batch_img[5],
                               model.inputs[6]: batch_img[6],
                               model.inputs[7]: batch_img[7],
                               model.inputs[8]: batch_img[8],
                               model.targets[0]: batch_mask,
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 10:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.inputs[3]: batch_img[3],
                               model.inputs[4]: batch_img[4],
                               model.inputs[5]: batch_img[5],
                               model.inputs[6]: batch_img[6],
                               model.inputs[7]: batch_img[7],
                               model.inputs[8]: batch_img[8],
                               model.inputs[9]: batch_img[9],
                               model.targets[0]: batch_mask,
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
        else:
            if len(batch_img) == 2 and len(batch_mask) == 2:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.targets[0]: batch_mask[0],
                               model.targets[1]: batch_mask[1],
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 3 and len(batch_mask) == 2:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.targets[0]: batch_mask[0],
                               model.targets[1]: batch_mask[1],
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 10 and len(batch_mask) == 2:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.inputs[3]: batch_img[3],
                               model.inputs[4]: batch_img[4],
                               model.inputs[5]: batch_img[5],
                               model.inputs[6]: batch_img[6],
                               model.inputs[7]: batch_img[7],
                               model.inputs[8]: batch_img[8],
                               model.inputs[9]: batch_img[9],
                               model.targets[0]: batch_mask[0],
                               model.targets[1]: batch_mask[1],
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 3 and len(batch_mask) == 3:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.targets[0]: batch_mask[0],
                               model.targets[1]: batch_mask[1],
                               model.targets[2]: batch_mask[2],
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                               model.sample_weights[2]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
    else:
        evaluated = []

    evaluated_all_weights = evaluated[:weights_len]
    evaluated_gradient_list = evaluated[weights_len:]

    for layer in layers:
        print("\nlayer #{} {}:{}".format(model.layers.index(layer), layer.__class__.__name__, layer.name) )
        print("input shape: {}".format(layer.input_shape) )
        print("output shape: {}".format(layer.output_shape) )

        for j, wt in enumerate(layer.trainable_weights):
            w = evaluated_all_weights[all_weights.index(wt)]
            wf = w.flatten()
            s = w.size
            print(" t_weights #{} {}: {} {} {} {}".format(j, w.shape,\
                wf[int(s//5)], wf[int(2*s//5)], wf[int(3*s//5)], wf[int(4*s//5)]) )

            g = evaluated_gradient_list[all_trainable_weights.index(wt)]
            gf = g.flatten()
            s = g.size
            print(" gradients #{} {}: {} {} {} {}".format(j, g.shape,\
                gf[int(s//5)], gf[int(2*s//5)], gf[int(3*s//5)], gf[int(4*s//5)]) )

        for j, wt in enumerate(layer.non_trainable_weights):
            w = evaluated_all_weights[all_weights.index(wt)]
            wf = w.flatten()
            s = w.size
            print(" nont_weights #{} {}: {} {} {} {}".format(j, w.shape,\
                wf[int(s//5)], wf[int(2*s//5)], wf[int(3*s//5)], wf[int(4*s//5)]) )



def save_model_output_gradients_wrt_to_input(model, batch_img, batch_mask_number, output_idx = None, save_path_prefix=None):
    if output_idx is None:
        gradient_list = K.gradients(model.outputs, model.inputs)
    else:
        gradient_list = K.gradients(model.outputs[output_idx], model.inputs)

    if not isinstance(batch_img, list):
        if batch_mask_number == 1:
            feed_dict={model.inputs[0]: batch_img,
                       model.sample_weights[0]: [1.0]*(batch_img.shape[0]),
                       K.learning_phase(): 1}
        elif batch_mask_number == 2:
            feed_dict={model.inputs[0]: batch_img,
                       model.sample_weights[0]: [1.0]*(batch_img.shape[0]),
                       model.sample_weights[1]: [1.0]*(batch_img.shape[0]),
                       K.learning_phase(): 1}
    elif len(batch_img) == 1 and batch_mask_number == 1:
        feed_dict={model.inputs[0]: batch_img[0],
                   model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                   K.learning_phase(): 1}
    elif len(batch_img) == 1 and batch_mask_number == 2:
        feed_dict={model.inputs[0]: batch_img[0],
                   model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                   model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                   K.learning_phase(): 1}
    elif len(batch_img) == 2 and batch_mask_number == 1:
        feed_dict={model.inputs[0]: batch_img[0],
                   model.inputs[1]: batch_img[1],
                   model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                   K.learning_phase(): 1}
    elif len(batch_img) == 2 and batch_mask_number == 2:
        feed_dict={model.inputs[0]: batch_img[0],
                   model.inputs[1]: batch_img[1],
                   model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                   model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                   K.learning_phase(): 1}
    elif len(batch_img) == 3 and batch_mask_number == 1:
        feed_dict={model.inputs[0]: batch_img[0],
                   model.inputs[1]: batch_img[1],
                   model.inputs[2]: batch_img[2],
                   model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                   K.learning_phase(): 1}
    elif len(batch_img) == 3 and batch_mask_number == 2:
        feed_dict={model.inputs[0]: batch_img[0],
                   model.inputs[1]: batch_img[1],
                   model.inputs[2]: batch_img[2],
                   model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                   model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                   K.learning_phase(): 1}
    elif len(batch_img) == 9 and batch_mask_number == 1:
        feed_dict={model.inputs[0]: batch_img[0],
                   model.inputs[1]: batch_img[1],
                   model.inputs[2]: batch_img[2],
                   model.inputs[3]: batch_img[3],
                   model.inputs[4]: batch_img[4],
                   model.inputs[5]: batch_img[5],
                   model.inputs[6]: batch_img[6],
                   model.inputs[7]: batch_img[7],
                   model.inputs[8]: batch_img[8],
                   model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                   K.learning_phase(): 1}
    elif len(batch_img) == 10 and batch_mask_number == 1:
        feed_dict={model.inputs[0]: batch_img[0],
                   model.inputs[1]: batch_img[1],
                   model.inputs[2]: batch_img[2],
                   model.inputs[3]: batch_img[3],
                   model.inputs[4]: batch_img[4],
                   model.inputs[5]: batch_img[5],
                   model.inputs[6]: batch_img[6],
                   model.inputs[7]: batch_img[7],
                   model.inputs[8]: batch_img[8],
                   model.inputs[9]: batch_img[9],
                   model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                   K.learning_phase(): 1}
    elif len(batch_img) == 10 and batch_mask_number == 2:
        feed_dict={model.inputs[0]: batch_img[0],
                   model.inputs[1]: batch_img[1],
                   model.inputs[2]: batch_img[2],
                   model.inputs[3]: batch_img[3],
                   model.inputs[4]: batch_img[4],
                   model.inputs[5]: batch_img[5],
                   model.inputs[6]: batch_img[6],
                   model.inputs[7]: batch_img[7],
                   model.inputs[8]: batch_img[8],
                   model.inputs[9]: batch_img[9],
                   model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                   model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                   K.learning_phase(): 1}
        
    
    evaluated_gradient_list = K.get_session().run(
         gradient_list, 
         feed_dict=feed_dict)

    print('the gradients of output w.r.t input')
    print(len(evaluated_gradient_list))
    print(evaluated_gradient_list[0].shape)

    # Save the gradients
    if save_path_prefix is not None:
        for i, eg in enumerate(evaluated_gradient_list):
            for j in range(eg.shape[0]):
                eg_j = eg[j, :]
                gradient_npy_path = save_path_prefix + 'input{}_batchIdx{}.npy'.format(str(i).zfill(2), str(j).zfill(2))
                np.save(gradient_npy_path, eg_j)

                max_val = eg_j.max()
                min_val = eg_j.min()
                eg_j = eg_j - min_val
                if (max_val - min_val) > 0:
                    eg_j = eg_j * 255.0 / (max_val - min_val)

                if len(eg_j.shape) == 3:
                    for k in range(eg_j.shape[2]):
                        eg_j_k = eg_j[:, :, k:(k+1)]
                        eg_j_k_img = array_to_img(eg_j_k, data_format=None, scale=False)
                        gradient_png_path = save_path_prefix + 'input{}_batchIdx{}_channel{}.png'.format(str(i).zfill(2), str(j).zfill(2), str(k).zfill(2))
                        eg_j_k_img.save(gradient_png_path)
                elif len(eg_j.shape) == 2:
                    for k in range(eg_j.shape[1]):
                        eg_j_k = np.expand_dims(eg_j[:, k:(k+1)], axis=0)
                        eg_j_k_img = array_to_img(eg_j_k, data_format=None, scale=False)
                        gradient_png_path = save_path_prefix + 'input{}_batchIdx{}_channel{}.png'.format(str(i).zfill(2), str(j).zfill(2), str(k).zfill(2))
                        eg_j_k_img.save(gradient_png_path)
                    




def handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def handle_dim_ordering2():
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CHANNEL_AXIS = 2
    else:
        CHANNEL_AXIS = 1


def get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


