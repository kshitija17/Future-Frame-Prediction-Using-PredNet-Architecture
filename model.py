# Import all dependencies
# %tensorflow_version 2.x
import tensorflow as tf
# tf.config.run_functions_eagerly(True)

import subprocess, os
import numpy as np
# import hickle as hkl
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.backend import hard_sigmoid
from tensorflow.keras.layers import UpSampling2D

# output_channels for each layer
layer_out_channels = (3, 48, 96, 192)


##### model #####

# ConvLSTM

class ConvLSTM(layers.Layer):
    """imprementation of Convolutional LSTM
    referring to
        https://github.com/takyamamoto/PredNet_Chainer/blob/master/network.py
        https://github.com/joisino/ConvLSTM/blob/master/network.py
        http://joisino.hatenablog.com/entry/2017/10/27/200000
        https://github.com/kn-lambda/Pred-Net/blob/master/model.py

    """

    def __init__(self, out_channels, kernel_size=3):
        super(ConvLSTM, self).__init__()

        # convolution settings
        # before fed into the first activation, channel size is enlarged to "out_channels"
        # and in after layers, channel size is not changed
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # to the first activation
        self.Wxc = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")
        self.Whc = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same", use_bias=False)
        # to input gate
        self.Wxi = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")
        self.Whi = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same", use_bias=False)
        # to forget gate
        self.Wxf = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")
        self.Whf = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same", use_bias=False)
        # to  output gate
        self.Wxo = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")
        self.Who = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same", use_bias=False)

        # peepholes
        # initialized when used for the first time
        self.Wci = None
        self.Wcf = None
        self.Wco = None

        # reset internal states
        self.reset_state()

    def reset_state(self, c=None, h=None):
        self.c = c
        self.h = h

    def _initialize_peephole(self, height, width):
        shape = (height, width, self.out_channels)
        self.Wci = tf.Variable(tf.zeros(shape))
        self.Wcf = tf.Variable(tf.zeros(shape))
        self.Wco = tf.Variable(tf.zeros(shape))

    def _initialize_state(self, x):
        # The following calculation is not elegant.
        # To initialize 'c', 'h', their 4-dim shape is required,
        # especially, the batch-size must be given which may be unknown when constructing the graph.
        # To avoid explicitly using batch-size, abstract input tenor 'x' is fed into the convolution
        # until we get the desired shape.
        tmp = self.Wxc(x)
        self.c = tf.zeros_like(tmp)
        self.h = tf.zeros_like(tmp)

    def __call__(self, x):
        """one-step execution of convLSTM
        Args:
            x : 4-dim (batch_size, height, width, channel) tensor
        """
        assert len(x.shape) == 4, "the dimension of the input tensor must be {}, but {}.".format(4, len(x.shape))

        if self.Wci is None:
            self._initialize_peephole(x.shape[1], x.shape[2])

        if self.c is None:
            self._initialize_state(x)

        ig = hard_sigmoid(self.Wxi(x) + self.Whi(self.h) + self.c * self.Wci)  # input gate
        fg = hard_sigmoid(self.Wxf(x) + self.Whf(self.h) + self.c * self.Wcf)  # forget gate
        new_c = fg * self.c + ig * tf.tanh(self.Wxc(x) + self.Whc(self.h))
        og = hard_sigmoid(self.Wxo(x) + self.Who(self.h) + new_c * self.Wco)  # output gate
        new_h = og * tf.tanh(new_c)

        self.c = new_c
        self.h = new_h

        return new_h


class R_block(layers.Layer):

    # Representation block in prednet

    def __init__(self, out_channels):
        super(R_block, self).__init__()
        # conv lstm layer
        self.convlstm = ConvLSTM(out_channels=out_channels)

        # output umsampled and passed to bottom layer R block
        self.up_sampling = UpSampling2D(size=(2, 2))

    def reset_state(self):
        self.convlstm.reset_state()

    def __call__(self, previous_R, previous_E, top_R=None):
        """
         previous_R: previous time(t-1) step value of R same layer
         previous_E: previous time(t-1) step value E same layer
         top_R: next layer R value in same time step(t)

         return value
         R for current time step(t)
        """

        if top_R is not None:
            up_R = self.up_sampling(top_R)
            lstm_input = tf.concat([previous_R, previous_E, up_R], axis=3)  # concat with channel dim

        else:
            lstm_input = tf.concat([previous_R, previous_E], axis=3)

        R = self.convlstm(lstm_input)
        return R


class E_block(layers.Layer):
    """
    prediction and error block

    """

    def __init__(self, out_channels, kernel_size=3, pixel_max=1.0, bottom=False):
        super(E_block, self).__init__()

        self.pixel_max = pixel_max
        self.bottom = bottom
        # down_sampling
        self.down_sampling = layers.MaxPool2D(pool_size=2, strides=2, padding="same")
        self.pred_conv = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")

        if bottom == False:
            self.target_conv = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")

    def __call__(self, R, bottom_E=None):
        """
        Args:
            bottom_E: bottom layer error in same time step
            R: current layer R value in same time step

        return value
        E : error value
        A_pred: generated prediction
        A_target: given target frame

        """

        if self.bottom == False:
            A_pred = tf.nn.relu(self.pred_conv(R))
            A_target = self.down_sampling(tf.nn.relu(self.target_conv(bottom_E)))
        else:
            # for bottom layer
            A_pred = tf.clip_by_value(tf.nn.relu(self.pred_conv(R)), 0.0, self.pixel_max)

            # input frame is sent to bottom_E

            if bottom_E is None:
                A_target = A_pred
            else:
                A_target = bottom_E

        # calculate error
        E = tf.concat([tf.nn.relu(A_pred - A_target), tf.nn.relu(A_target - A_pred)], axis=3)
        return E, A_pred


class prednet(tf.keras.Model):
    """
    prednet model structure

    """

    def __init__(self, out_channels=layer_out_channels):
        super(prednet, self).__init__()
        """
        layer_
        out_channels: tuple contains number of channels for each layer
        """
        self.out_channels = out_channels
        # total layers
        self.num_layers = len(out_channels)

        for layer in range(self.num_layers):
            setattr(self, "R_block" + str(layer), R_block(out_channels=out_channels[layer]))

            if layer == 0:
                setattr(self, "E_block" + str(layer), E_block(out_channels[layer], bottom=True))

            else:
                setattr(self, "E_block" + str(layer), E_block(out_channels[layer]))

        self.reset_state()

    def reset_state(self):
        # reset value for each layer
        self.E = None
        self.R = None

    def _one_step(self, x):

        """
        R is updated top to bottom
        next frame is predicted and comapred with ground truth input frame
        Next, E is updated bottom to top

        args
        x : 4-dim(batch_size, width,height,channels)

        """

        if x is not None:
            assert len(x.shape) == 4, "input tensor dimension should be {}, but {} is given.".format(4, len(x.shape))

        if self.E is None:
            self.E = []
            self.R = []

            temp_A = tf.zeros_like(x, tf.float32)

            for layer in range(self.num_layers):
                """

                double the number of channels

                """
                double_channels = layers.Conv2D(filters=2 * self.out_channels[layer], kernel_size=1, trainable=False,
                                                kernel_initializer=tf.zeros_initializer())

                # shape of E is twice the shape of A
                temp_E = double_channels(temp_A)

                self.E.append(tf.zeros_like(temp_E))

                # shape of R = shape of A
                self.R.append(tf.zeros_like(temp_A))

                getattr(self, "R_block" + str(layer)).reset_state()

                if layer != self.num_layers - 1:
                    # change the size of temp A for next layer
                    next_layer_dim = layers.Conv2D(filters=self.out_channels[layer + 1], kernel_size=2, strides=2,
                                                   trainable=False, kernel_initializer=tf.zeros_initializer())
                    temp_A = next_layer_dim(temp_A)

        # top down calculations for R block

        for layer in reversed(range(self.num_layers)):
            if layer != self.num_layers - 1:
                updated_R = getattr(self, "R_block" + str(layer))(self.R[layer], self.E[layer], self.R[layer + 1])
            else:
                updated_R = getattr(self, "R_block" + str(layer))(self.R[layer], self.E[layer])

            self.R[layer] = updated_R

        # bottom up calculation for E block

        for layer in range(self.num_layers - 1):
            if layer != 0:
                updated_E, _ = getattr(self, "E_block" + str(layer))(self.R[layer], self.E[layer - 1])

            else:
                updated_E, pred = getattr(self, "E_block" + str(layer))(self.R[layer], x)

            self.E[layer] = updated_E

        temp_loss = tf.reduce_mean(self.E[0])  # consider loss for this time step

        return temp_loss, pred

    @tf.function
    def __call__(self, x, num_pred_frames=0):
        """
        Args
        x :  5 dimension (batch_size,time_step, width, height, channels)
        num_pred_frames: number of future predicted frames
        """

        assert len(x.shape) == 5, "the dimension of the input tensor must be{}, but{} is given.".format(5, len(x.shape))

        time_steps = x.shape[1]

        total_loss = 0
        pred_frame_list = []
        self.reset_state()

        # predict for each time step
        for t in range(time_steps + num_pred_frames):

            if t < time_steps:
                x_t = x[:, t, :, :, :]
            else:
                x_t = None

            loss, pred = self._one_step(x_t)
            pred_frame_list.append(pred)

            if t > 0 and t < time_steps:
                total_loss += loss

        total_loss = total_loss / (time_steps - 1)

        return total_loss, pred_frame_list
