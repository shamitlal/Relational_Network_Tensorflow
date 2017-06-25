import argparse
import os
import scipy.misc
import numpy as np
from glob import glob
from utils import load_data

from model import relationalNetwork
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')##
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--dataset_name', dest='dataset_name', default='data', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')  ##
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--load_size', dest='load_size', type=int, default=75, help='scale images to this size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')  ##
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session() as sess:
        model = relationalNetwork(sess, image_size=args.load_size, batch_size=args.batch_size,
                        dataset_name=args.dataset_name, checkpoint_dir=args.checkpoint_dir)

        if args.phase == 'train':
            model.train(args)
        else:
            model.test(args)

if __name__ == '__main__':
    # print args.dataset_name
    # data = glob('./datasets/{}/train/*.jpg'.format(args.dataset_name))
    # batch = []
    # for file in data:
    #     img_ab = load_data(file)
    #     print img_ab.shape
        # batch.append(load_data(file))
    # batch = [load_data(batch_file) for batch_file in batch_files]
    tf.app.run()