import tensorflow as tf
import numpy as np
from .network import Network
from ..fcn.config import cfg

class VGGnet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
        n_classes = cfg.NCLASSES
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
         .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
         .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool5'))

        (self.feed('pool5')
         .conv(7, 7, 4096, 1, 1, name='fc6')
         .dropout(0.5, name='drop6')
         .conv(1, 1, 4096, 1, 1, name='fc7')
         .dropout(0.5, name='drop7')
         .conv(1, 1, n_classes, 1, 1, relu=False,name='fc8'))

        (self.feed('fc8')
         .upsample(2, 32, name='upsample_fc8'))
