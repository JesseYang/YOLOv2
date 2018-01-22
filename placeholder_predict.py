import os
import numpy as np
import argparse

import tensorflow as tf
from tensorpack import *

from train_lite import Model

def predict(args):


    with tf.Session() as sess:

        model = Model(data_format='NHWC')
        image = tf.placeholder(dtype=tf.float32,
                               shape=[1, 416, 416, 3],
                               name='input_image')
        tx = tf.placeholder(dtype=tf.float32,
                            shape=[1, 5, 1, 13, 13],
                            name='input_tx')
        ty = tf.placeholder(dtype=tf.float32,
                            shape=[1, 5, 1, 13, 13],
                            name='input_ty')
        tw = tf.placeholder(dtype=tf.float32,
                            shape=[1, 5, 1, 13, 13],
                            name='input_tw')
        th = tf.placeholder(dtype=tf.float32,
                            shape=[1, 5, 1, 13, 13],
                            name='input_th')
        tprob = tf.placeholder(dtype=tf.float32,
                               shape=[1, 5, 20, 13, 13],
                               name='input_tprob')
        spec_mask = tf.placeholder(dtype=tf.bool,
                                   shape=[1, 5, 13, 13],
                                   name='spec_mask')
        truth_box = tf.placeholder(dtype=tf.float32,
                                   shape=[1, 30, 4],
                                   name='truth_box')
        ori_shape = tf.placeholder(dtype=tf.float32,
                                   shape=[1, 3],
                                   name='ori_shape')

        with TowerContext('', is_training=False):
            model.build_graph(image, tx, ty, tw, th, tprob, spec_mask, truth_box, ori_shape)

        saver = tf.train.Saver()
        saver.restore(sess, args.model)

        output = tf.get_default_graph().get_tensor_by_name("lite_output:0")
        output_ary = sess.run(output, feed_dict={image: np.zeros((1, 416, 416, 3))})

        print(output_ary.shape)

        tf.train.write_graph(sess.graph_def, "./", "yolo.pb", as_text=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to the model file', required=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    args = parser.parse_args()
    predict(args)
