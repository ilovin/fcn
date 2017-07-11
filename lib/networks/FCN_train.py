import tensorflow as tf
from .network import Network
from ..fcn.config import cfg

class FCN_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.label = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='label')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({ 'data':self.data,'label':self.label})
        self.trainable = trainable
        self.setup()

    def setup(self):
        n_classes = cfg.NCLASSES
        (self.feed('data')
             .conv_norm(5, 5, 32, 1, 1, name='conv1', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv_norm(3, 3, 64, 1, 1, name='conv2_1', trainable=False)
             .conv_norm(3, 3, 64, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv_norm(3, 3, 128, 1, 1, name='conv3_1', trainable=False)
             .conv_norm(3, 3, 128, 1, 1, name='conv3_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv_norm(3, 3, 256, 1, 1, name='conv4_1')
             .conv_norm(3, 3, 256, 1, 1, name='conv4_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv_norm(3, 3, 512, 1, 1, name='conv5_1')
             .conv_norm(3, 3, 512, 1, 1, name='conv5_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool5')
             .conv_norm(3, 3, 512, 1, 1, name='conv6_1')
             .conv_norm(3, 3, 512, 1, 1, name='conv6_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool6'))
        '''
        (self.feed('pool3')
             .conv_norm(1, 1, 2, 1, 1, name='conv_pool3'))
        (self.feed('pool4')
             .conv_norm(1, 1, 2, 1, 1, name='conv_pool4'))
        (self.feed('pool5')
             .conv_norm(1, 1, 2, 1, 1, name='conv_pool5'))
        (self.feed('pool6')
             .conv_norm(1, 1, 2, 1, 1, name='conv_pool6'))


        (self.feed('conv_pool6')
         .upsample(2, 2, name='upsample6'))
        (self.feed('conv_pool5','upsample6')
         .add(name = 'concat5'))
        (self.feed('concat5')
         .upsample(2, 2, name='upsample5'))
        (self.feed('conv_pool4','upsample5')
         .add(name = 'concat4'))
        (self.feed('concat4')
         .upsample(2, 2, name='upsample4'))
        (self.feed('conv_pool3','upsample4')
         .add(name = 'concat3'))
        (self.feed('concat3')
         .upsample(2, 8, name='output_logits'))
        '''

        (self.feed('pool3')
         .conv_norm(1, 1, 128, 1, 1, name='conv_pool3'))
        (self.feed('pool4')
         .conv_norm(1, 1, 128, 1, 1, name='conv_pool4'))
        (self.feed('pool5')
         .conv_norm(1, 1, 128, 1, 1, name='conv_pool5'))
        (self.feed('pool6')
         .conv_norm(1, 1, 128, 1, 1, name='conv_pool6'))

        (self.feed('conv_pool6')
         .upsample(128, 2, name='upsample6'))
        (self.feed('conv_pool5', 'upsample6')
         .add(name='concat5'))
        (self.feed('concat5')
         .upsample(128, 2, name='upsample5'))
        (self.feed('conv_pool4', 'upsample5')
         .add(name='concat4'))
        (self.feed('concat4')
         .upsample(128, 2, name='upsample4'))
        (self.feed('conv_pool3', 'upsample4')
         .add(name='concat3'))
        (self.feed('concat3')
         .upsample(128, 8, name='upsample3'))
        (self.feed('upsample3')
         .conv_final(1, 1, 2, 1, 1,name = 'output_logits'))

if __name__ == '__main__':
    network = FCN_train()
    print('Done')
