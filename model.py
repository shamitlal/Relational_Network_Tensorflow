from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from skimage import color

from ops import *
from utils import *

class relationalNetwork(object):
    def __init__(self, sess, image_size=75,
                 batch_size=8, dataset_name='data',
                 checkpoint_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            convf_dim: (optional) Dimension of filters in first conv layer. [64]
            g_fc: size of fully connected layer
        """
        self.sess = sess
        self.is_grayscale = 0  #Images are colored
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = 3
        self.convf_dim = 24
        self.g_fc = 256
        self.qst_len = 11
        # batch normalization : deals with poor initialization helps gradient flow
        self.bn1 = batch_norm(name='bn1')
        self.bn2 = batch_norm(name='bn2')
        self.bn3 = batch_norm(name='bn3')
        self.bn4 = batch_norm(name='bn4')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.input_img = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.num_channels],
                                        name='images')
        self.input_qst = tf.placeholder(tf.float32,
                                        [self.batch_size, self.qst_len],
                                        name="question")

        self.input_label = tf.placeholder(tf.int64,
                                        [self.batch_size],
                                        name="labels")

        self.keep_prob = tf.placeholder(tf.float32)   #set it to 0.5
        # print self.real_A.get_shape(), self.real_B.get_shape()

        self.output_label = self.forward(self.input_img, self.input_qst)
        self.input_imp_sum = tf.summary.image("input_img", self.input_img, max_outputs=self.batch_size)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output_label, labels=self.input_label))
                                                                            

        self.correct_prediction = tf.equal(tf.argmax(self.output_label, 1), self.input_label)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.loss_sum = tf.summary.scalar("loss", self.loss)
        self.saver = tf.train.Saver()

    def train(self, args):
        """Train Relational network"""

        optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.loss)
        
        # tf.initialize_all_variables().run()
        self.sess.run(tf.global_variables_initializer())

        self.sum = tf.summary.merge([self.input_imp_sum, self.loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        rel, norel = load_data("train")
        # batch_idxs = (len(rel)+len(norel))// self.batch_size
        batch_idxs = (len(rel)+len(norel))// self.batch_size  # training only for relational dataset
        print "len of rel 1=",batch_idxs
        for epoch in xrange(args.epoch):

            
            print "batch size=",batch_idxs
            random.shuffle(rel)
            random.shuffle(norel)

            rel_norel=rel+norel  ####
            random.shuffle(rel_norel)####

            print "len of rel 2=",len(rel_norel)
            rel_norel_tuple = cvt_data_axis(rel_norel) ####

            # rel_norel_tuple = cvt_data_axis(rel_norel)
            # norel_tuple = cvt_data_axis(norel)
            print "len of rel 3=",len(rel)
            for idx in xrange(0, batch_idxs):
                print "len of rel 4=",len(rel_norel)
                img, qst, ans = tensor_data(rel_norel_tuple, idx, self.batch_size) ####
                print "len of rel_norel 5=",len(rel_norel)

                print "Batch images shape", img.shape
                print "Batch question shape", qst.shape
                print "Batch answer shape", ans.shape

                #update the relational network
                _, summary_str = self.sess.run([optim, self.sum], feed_dict={ self.input_img: img, self.input_qst: qst, 
                                                                self.input_label: ans, self.keep_prob: 0.5 })
                self.writer.add_summary(summary_str, counter)

                

                train_loss = self.loss.eval({ self.input_img: img, self.input_qst: qst, 
                                                self.input_label: ans, self.keep_prob: 0.5 })

                train_accuracy = self.accuracy.eval({ self.input_img: img, self.input_qst: qst, 
                                                self.input_label: ans, self.keep_prob: 0.5 })
                
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, train_loss, train_accuracy))

                if np.mod(counter, 50) == 2:
                    self.save(args.checkpoint_dir, counter)

            # rel_test, norel_test = load_data("test")
            # rel_norel_test=rel_test+norel_test

    def cvt_coord(self, i):
        return [(i/5-2)/2., (i%5-2)/2.]
    
    def forward(self, image, qst, y=None):
        
        coord_list = [(np.array([self.cvt_coord(i) for _ in range(self.batch_size)])) for i in range(25)]
        # image is (75 x 75 x 3)
        e1 = self.bn1(conv2d(image, self.convf_dim, name='e1_conv'))
        # e1 is (38 x 28 x self.convf_dim)
        e2 = self.bn2(conv2d(lrelu(e1), self.convf_dim, name='e2_conv'))
        # e2 is (19 x 19 x self.convf_dim)
        e3 = self.bn3(conv2d(lrelu(e2), self.convf_dim, name='e3_conv'))
        # e3 is (10 x 10 x self.convf_dim)
        e4 = self.bn4(conv2d(lrelu(e3), self.convf_dim, name='e4_conv'))
        # e4 is (5 x 5 x self.convf_dim)
        
        x_g = 0
        reuse_flag = False
        for i in range(25):
            fir = e4[:,i//5,i%5,:]
            fir = tf.concat([fir, coord_list[i]], 1)
            for j in range(25):
                sec = e4[:,j//5,j%5,:]
                sec = tf.concat([sec, coord_list[j]], 1)
                x_ = tf.concat([fir,sec,qst],1) # size of x=(24+2)*2+11
                g1 = lrelu(linear(x_, self.g_fc, "g1_fc", reuse=reuse_flag))
                g2 = lrelu(linear(g1, self.g_fc, "g2_fc", reuse=reuse_flag))
                g3 = lrelu(linear(g2, self.g_fc, "g3_fc", reuse=reuse_flag))
                g4 = lrelu(linear(g3, self.g_fc, "g4_fc", reuse=reuse_flag))
                x_g += g4
                reuse_flag = True

        f1 = lrelu(linear(x_g, self.g_fc, "f1_fc"))
        f2 = lrelu(linear(f1, self.g_fc, "f2_fc"))
        f2_drop = tf.nn.dropout(f2, self.keep_prob)
        f3 = linear(f2_drop, 10, "f3_fc")
        return f3

    
    def save(self, checkpoint_dir, step):
        model_name = "relationalNetwork.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    