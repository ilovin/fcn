# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from .network import Network
from ..fcn.config import cfg


class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name='annotation')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data,'annotation':self.annotation})
        self.trainable = trainable
        self.setup()

    def setup(self):
        n_classes = cfg.NCLASSES
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1')
         .conv(3, 3, 64, 1, 1, name='conv1_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1')
         .conv(3, 3, 128, 1, 1, name='conv2_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool5'))

        (self.feed('pool5')
         .conv(7, 7, 4096, 1, 1, name='fc6')
         .dropout(self.keep_prob, name='drop6')
         .conv(1, 1, 4096, 1, 1, name='fc7')
         .dropout(self.keep_prob, name='drop7')
         .conv(1, 1, n_classes, 1, 1, name='fc8',relu=False))

        (self.feed('fc8')
         .upsample(n_classes, 32, name='upsample_fc8'))

        # (self.feed('fc8')
        #  .upconv(shape=None,c_o=n_classes, stride=32, name='upsample_fc8'))
