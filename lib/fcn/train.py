import numpy as np
import os
import tensorflow as tf
from ..fcn.config import cfg
from lib.fcn.utils.timer import Timer
from lib.fcn.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from lib.fcn.utils.training import get_valid_logits_and_labels
from lib.fcn.utils.augmentation import (distort_randomly_image_color,crop_or_resize_to_fixed_size_and_rotate_output,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)
from lib.fcn.utils.vgg_preprocessing import  _R_MEAN, _G_MEAN, _B_MEAN

class SolverWrapper(object):
    def __init__(self, sess, network, imgdb, pre_train,output_dir, logdir):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imgdb = imgdb
        self.pre_train=pre_train
        self.output_dir = output_dir
        print('done')
        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = tf.summary.FileWriter(logdir=logdir,
                                             graph=tf.get_default_graph(),
                                             flush_secs=5)

    def snapshot(self, sess, iter):
        net = self.net
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        
        if self.pre_train.split('/')[-1].split('.')[0]=='32s':
            filename = (cfg.TRAIN.SNAPSHOT_PREFIX + '_16'+infix +
                        '_iter_{:d}'.format(iter + 1) + '.ckpt')
        elif self.pre_train.split('/')[-1].split('.')[0]=='16s':
            filename = (cfg.TRAIN.SNAPSHOT_PREFIX + '_8' + infix +
                        '_iter_{:d}'.format(iter + 1) + '.ckpt')
        else:
            filename = (cfg.TRAIN.SNAPSHOT_PREFIX + '_32' + infix +
                        '_iter_{:d}'.format(iter + 1) + '.ckpt')
        
        #filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
         #           '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))


    def train_model(self, sess, max_iters, restore=False):
        image_size = cfg.IMG_SHAPE
        tfrecord_filename = self.imgdb.path



        filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=100)
        image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)
        resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image,
                                                                                                        annotation,
                                                                                                        image_size,
                                                                                                        mask_out_number=255)
        image_batch, annotation_batch = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                               batch_size=1,
                                                               capacity=500,
                                                               num_threads=2,
                                                               min_after_dequeue=200)

        loss=self.net.build_loss()
        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()

        # optimizer
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        else:
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step = tf.Variable(0, trainable=False)
        with_clip = False
        if with_clip:
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = opt.minimize(loss, global_step=global_step)

        # intialize variables
        local_vars_init_op = tf.local_variables_initializer()
        global_vars_init_op = tf.global_variables_initializer()

        combined_op = tf.group(local_vars_init_op, global_vars_init_op)
        sess.run(combined_op)
        restore_iter = 1

        # load vgg16
        if not restore:
            print('Loading pretrained model ')
            self.net.load(self.pre_train, sess, True)
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
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                for iter in range(restore_iter, max_iters):
                    timer.tic()
                    # learning rate
                    if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                        sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))

                    # get one batch
                    img_batch,annotate_batch = sess.run([image_batch,annotation_batch])
                    # Subtract the mean pixel value from each pixel
                    img_batch = img_batch - [_R_MEAN, _G_MEAN, _B_MEAN]
                    img_batch = np.reshape(img_batch,[1]+image_size+[3])
                    feed_dict = {
                        self.net.data:img_batch,
                        self.net.label: annotate_batch,
                        self.net.keep_prob: 0.5
                    }

                    fetch_list = [loss,summary_op,train_op]
                    loss_cls,summary_str, _ =  sess.run(fetches=fetch_list, feed_dict=feed_dict)

                    self.writer.add_summary(summary=summary_str, global_step=global_step.eval())
                    _diff_time = timer.toc(average=False)

                    if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                        print('iter: %d / %d, total loss: %.7f, lr: %.7f'%\
                                (iter, max_iters, loss_cls ,lr.eval()))
                        print('speed: {:.3f}s / iter'.format(_diff_time))
                    if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                        self.snapshot(sess, iter)
                iter = max_iters-1
                self.snapshot(sess, iter)
                coord.request_stop()
        except tf.errors.OutOfRangeError:
            print('finish')
        finally:
            coord.request_stop()
        coord.join(threads)


def train_net(network, imgdb, pre_train,output_dir, log_dir, max_iters=40000, restore=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imgdb, pre_train,output_dir, logdir= log_dir)
        print('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print('done solving')
