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

class pix2pix(object):
    def __init__(self, sess, image_size=75,
                 batch_size=4, dataset_name='facades',
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

        self.input_label = tf.placeholder(tf.float32,
                                        [self.batch_size],
                                        name="labels")

        self.keep_prob = tf.placeholder(tf.float32)   #set it to 0.5
        # print self.real_A.get_shape(), self.real_B.get_shape()

        self.output_label = self.forward(self.input_img, self.input_qst)
        self.input_imp_sum = tf.image_summary("input_img", self.input_img, max_images=self.batch_size)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.output_label, self.input_label))

        self.correct_prediction = tf.equal(tf.argmax(self.output_label, 1), self.input_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.loss_sum = tf.scalar_summary("loss", self.loss)
        self.saver = tf.train.Saver()


    def load_random_samples(self):
        data = np.random.choice(glob('./datasets/{}/val/*/*.jpg'.format(self.dataset_name)), self.batch_size)
        sample = [load_data(sample_file) for sample_file in data]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )

        # sample images - normalized lab
        sample_images_l = sample_images[:,:,:,:1]
        sample_images_l = (sample_images_l + 1.) / 2.

        samples = np.concatenate((sample_images_l, samples), 3)
        rgb_samples = np.asarray([color.lab2rgb(sample) for sample in samples])
        save_images(rgb_samples, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train Relational network"""

        optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.loss)
        
        # tf.initialize_all_variables().run()
        self.sess.run(tf.global_variables_initializer())

        self.sum = tf.merge_summary([self.input_imp_sum, self.loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        rel, norel = load_data("train")

        for epoch in xrange(args.epoch):

            batch_idxs = min(len(rel)+len(norel), args.train_size) // self.batch_size
            
            random.shuffle(rel)
            random.shuffle(norel)

            rel = cvt_data_axis(rel)
            norel = cvt_data_axis(norel)

            for idx in xrange(0, batch_idxs):

                img, qst, ans = tensor_data(rel, idx, self.batch_size)
               
                # Update D network
                print "Batch images shape", img.shape
                print "Batch question shape", qst.shape
                print "Batch answer shape", ans.shape

                #update the relational network
                _, summary_str = self.sess.run([optim, self.sum], feed_dict={ self.input_img: img, self.input_qst: qst, 
                                                                self.input_label, ans, self.keep_prob: 0.5 })
                self.writer.add_summary(summary_str, counter)

                

                train_loss = self.loss.eval({ self.input_img: img, self.input_qst: qst, 
                                                self.input_label, ans, self.keep_prob: 0.5 })

                train_accuracy = self.accuracy.eval({ self.input_img: img, self.input_qst: qst, 
                                                self.input_label, ans, self.keep_prob: 0.5 })
                
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, train_loss, train_accuracy))

                if np.mod(counter, 3) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 250) == 2:
                    self.save(args.checkpoint_dir, counter)

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
        reuse_flag = 0
        for i in range(25):
            fir = e4[:,i/5,i%5,:]
            fir = tf.concat([fir, coord_list[i]], 1)
            for j in range(25):
                sec = e4[:,j/5,j%5,:]
                sec = tf.concat([sec, coord_list[j]], 1)
                x_ = tf.concat([fir,sec,qst],1) # size of x=(24+2)*2+11
                g1 = lrelu(linear(x_, self.g_fc, "g1_fc", reuse=reuse_flag))
                g2 = lrelu(linear(g1, self.g_fc, "g2_fc", reuse=reuse_flag))
                g3 = lrelu(linear(g2, self.g_fc, "g3_fc", reuse=reuse_flag))
                g4 = lrelu(linear(g3, self.g_fc, "g4_fc", reuse=reuse_flag))
                x_g += g4
                reuse_flag=1

        f1 = lrelu(linear(x_g, self.g_fc, "f1_fc"))
        f2 = lrelu(linear(f1, self.g_fc, "f2_fc"))
        f2_drop = tf.nn.dropout(f2, self.keep_prob)
        f3 = linear(f2_drop, 10, "f3_fc"))
        return f3

    def sampler(self, image, qst, y=None):
        tf.get_variable_scope().reuse_variables()

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
        reuse_flag = 1
        for i in range(25):
            fir = e4[:,i/5,i%5,:]
            fir = tf.concat([fir, coord_list[i]], 1)
            for j in range(25):
                sec = e4[:,j/5,j%5,:]
                sec = tf.concat([sec, coord_list[j]], 1)
                x_ = tf.concat([fir,sec,qst],1) # size of x=(24+2)*2+11
                g1 = lrelu(linear(x_, self.g_fc, "g1_fc", reuse=reuse_flag))
                g2 = lrelu(linear(g1, self.g_fc, "g2_fc", reuse=reuse_flag))
                g3 = lrelu(linear(g2, self.g_fc, "g3_fc", reuse=reuse_flag))
                g4 = lrelu(linear(g3, self.g_fc, "g4_fc", reuse=reuse_flag))
                x_g += g4
                reuse_flag=1

        f1 = lrelu(linear(x_g, self.g_fc, "f1_fc"))
        f2 = lrelu(linear(f1, self.g_fc, "f2_fc"))
        f2_drop = tf.nn.dropout(f2, self.keep_prob)
        f3 = linear(f2_drop, 10, "f3_fc"))
        return f3

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        tf.initialize_all_variables().run()

        sample_files = glob('./datasets/{}/val/*/*.jpg'.format(self.dataset_name))

        # sort testing input
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, is_test=True) for sample_file in sample_files]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)
        else:
            sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images):
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            save_images(samples, [self.batch_size, 1],
                        './{}/test_{:04d}.png'.format(args.test_dir, idx))