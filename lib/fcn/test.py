import sys
import os,shutil
import collections
import numpy as np
import os
import tensorflow as tf
import cv2
from lib.fcn.utils.timer import Timer
from ..fcn.config import cfg

class SolverWrapper(object):
    def __init__(self, sess, network, imgdb, output_dir, logdir, pretrained_model=None):
        self.net = network
        self.imgdb = imgdb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        print('done')

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = tf.summary.FileWriter(logdir=logdir,
                                             graph=tf.get_default_graph(),
                                             flush_secs=5)


    def test_model(self,sess,testDir=None,restore = True):
        img_size = cfg.IMG_SHAPE
        oneQuarterSize = [int(img_size[0]/4),int(img_size[1]/4)]

        global_step = tf.Variable(0, trainable=False)
        # intialize variables
        local_vars_init_op = tf.local_variables_initializer()
        global_vars_init_op = tf.global_variables_initializer()

        combined_op = tf.group(local_vars_init_op, global_vars_init_op)
        sess.run(combined_op)
        # resuming a trainer
        if restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.output_dir)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(sess, tf.train.latest_checkpoint(self.output_dir))
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_iter))
                print('done')
            except:
                raise Exception('Check your pretrained {:s}'.format(ckpt.model_checkpoint_path))

        timer = Timer()

        upsampled_batch = self.net.get_output('output_logits')
        saveDir = 'data/result'
        if os.path.exists(saveDir):
            shutil.rmtree(saveDir)
        os.makedirs(saveDir)
        for file in os.listdir(testDir):
            timer.tic()

            img = cv2.imread(os.path.join(testDir,file),1)
            img = cv2.resize(img,tuple(img_size))
            print(file,end=' ')

            # get one batch
            img = np.reshape(img, [1]+img_size+[3])
            feed_dict = {
                self.net.data: img,
                self.net.keep_prob: 1.0
            }
            fetch_list = [upsampled_batch]
            output = sess.run(fetches=fetch_list, feed_dict=feed_dict)
            output = np.squeeze(output)
            output = np.argmax(output,axis=2)

            _diff_time = timer.toc(average=False)
            output=output*127
            img = np.reshape(img, img_size+[3])
            result=np.array(img,dtype=np.int32)
            result[:,:,1]=result[:,:,1]+output
            cv2.imwrite(os.path.join(saveDir,file.split('.')[0]+'.png'),result)
            print('cost time: {:.3f}'.format(_diff_time))
            #visualize_segmentation_adaptive(np.array(output),cls_dict)


def test_net(network, imgdb, testDir, output_dir, log_dir, pretrained_model=None,restore=True):
    """Train a Fast R-CNN network."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imgdb, output_dir, logdir= log_dir, pretrained_model=pretrained_model)
        print('Solving...')
        sw.test_model(sess, testDir=testDir, restore=restore)
        print('done solving')

