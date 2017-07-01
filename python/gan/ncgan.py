import sys
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_shape(tensor):
    return tensor.get_shape().as_list()

def get_vars(modeldict):
    print(list(filter(lambda k: k.startswith('W_'), modeldict.keys())))
    return [modeldict[v] for v in filter(lambda k: k.startswith('W_'), modeldict.keys())]

def lkrelu(x, a=0.0):
    return tf.maximum(a * x, x)

def generator(zs, ys, G_wdim):
    G = dict()
    # Generator
    with tf.name_scope('G'):
        with tf.name_scope('concat_zy'):
            G['c_zy'] = tf.concat([zs, ys], 1)

        with tf.name_scope('L_1'):
            G['W_1'] = tf.Variable(tf.truncated_normal([get_shape(G['c_zy'])[-1], G_wdim['W_1']], stddev=0.02))
            G['z_1'] = tf.matmul(G['c_zy'], G['W_1'])
            G['a_1'] = tf.nn.dropout(lkrelu(G['z_1']), 1.0)

        with tf.name_scope('L_o'):
            G['W_o'] = tf.Variable(tf.truncated_normal([get_shape(G['a_1'])[-1], G_wdim['W_o']], stddev=0.02))
            G['z_o'] = tf.matmul(G['a_1'], G['W_o'])
            G['a_o'] = tf.nn.tanh(G['z_o']) #tanh
    return G

def discriminator(real_tensor, fake_tensor, D_wdim):
    D = dict()
    real_xs, real_ys = real_tensor
    fake_xs, fake_ys = fake_tensor
    # Discriminator
    with tf.name_scope('D'):
        with tf.name_scope('concat_xy'):
            D['rc_xy'] = tf.concat([real_xs, real_ys], 1)
            D['fc_xy'] = tf.concat([fake_xs, fake_ys], 1)

        with tf.name_scope('L_1'):
            D['W_1'] = tf.Variable(tf.truncated_normal([get_shape(D['rc_xy'])[-1], D_wdim['W_1']], stddev=0.02))
            D['rz_1'] = tf.matmul(D['rc_xy'], D['W_1'])
            D['ra_1'] = tf.nn.dropout(lkrelu(D['rz_1']), 1.0)
            D['fz_1'] = tf.matmul(D['fc_xy'], D['W_1'])
            D['fa_1'] = tf.nn.dropout(lkrelu(D['fz_1']), 1.0)

        with tf.name_scope('L_o'):
            D['W_o'] = tf.Variable(tf.truncated_normal([get_shape(D['ra_1'])[-1], D_wdim['W_o']], stddev=0.02))
            D['rz_o'] = tf.matmul(D['ra_1'], D['W_o'])
            D['ra_o'] = tf.nn.sigmoid(D['rz_o'])
            D['fz_o'] = tf.matmul(D['fa_1'], D['W_o'])
            D['fa_o'] = tf.nn.sigmoid(D['fz_o'])


        tf.summary.image('rdenimg', tf.reshape(real_xs , [-1, 28, 28, 1]), 10)
        tf.summary.image('fdenimg', tf.reshape(fake_xs , [-1, 28, 28, 1]), 10)
    return D

def viz_gen_class(genimgs_tensor):
    imgs = tf.reshape(genimgs_tensor, [-1, 28, 28, 1])
    with tf.name_scope('genclass'):
        summary = tf.summary.merge([tf.summary.image('c%d' % i, tf.expand_dims(tf.gather(imgs, i), 0)) for i in range(10)], collections='genclass')
    return summary


def random_one_hot(size):
    one_hots = np.zeros(size)
    one_hots[np.arange(size[0]), np.random.randint(size[1], size=size[0])] = 1
    return one_hots

logdir = sys.argv[1]

#clean log
for filename in os.listdir(logdir):
    os.remove(os.path.join(logdir, filename))

batch_size = 64
zdim = 100
step_epochs = 100000
k = 1
lr = 0.005

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

G_wdim = {
    'W_1' : 128,
    'W_o' : 784
}

D_wdim = {
    'W_1' : 128,
    'W_o' : 1
}

with tf.device('/gpu:0'):

    zs = tf.placeholder(tf.float32, [None, zdim]) # sampled from pz(z)
    g_ys = tf.placeholder(tf.float32, [None, 10])
    G = generator(zs, g_ys, G_wdim)

    real_xs = tf.placeholder(tf.float32, [None, 28*28])
    real_ys = tf.placeholder(tf.float32, [None, 10])
    fake_xs = G['a_o']
    fake_ys = g_ys

    D = discriminator((real_xs, real_ys), (fake_xs, fake_ys), D_wdim)

    with tf.name_scope('G'):
        G_loss = -tf.reduce_mean(tf.log(D['fa_o']))
        tf.summary.scalar('loss', G_loss)

    with tf.name_scope('D'):
        D_loss = -(tf.reduce_mean(tf.log(D['ra_o']) + tf.log(1.0 - D['fa_o'])))
        tf.summary.scalar('loss', D_loss)

    G_train_step = tf.train.AdamOptimizer().minimize(G_loss, var_list=get_vars(G))
    D_train_step = tf.train.AdamOptimizer().minimize(D_loss, var_list=get_vars(D))

    genimg_summary = viz_gen_class(G['a_o'])
    total_summary = tf.summary.merge_all()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logdir, sess.graph)
    summary_nb = 0
    for epoch in range(step_epochs):
        for step in range(k):
            mini_batch_xs, mini_batch_ys = mnist.train.next_batch(batch_size, shuffle=True)
            _, dloss, summary = sess.run([D_train_step, D_loss, total_summary],
                                        feed_dict={real_xs: 2. * mini_batch_xs - 1. , real_ys: mini_batch_ys, zs:np.random.uniform(-1, 1, (batch_size, zdim)), g_ys: mini_batch_ys })
            writer.add_summary(summary, summary_nb)
            summary_nb += 1
            print('D loss, epoch %d step %d value %f' % (epoch, step, dloss))

        _, gloss, tb = sess.run([G_train_step, G_loss, G['a_o']], feed_dict={zs:np.random.uniform(-1, 1, (batch_size, zdim)), g_ys: random_one_hot((batch_size, 10))})
        imgs_summary = sess.run(genimg_summary, feed_dict={zs:np.random.uniform(-1, 1, (10, zdim)), g_ys: np.eye(10)})
        writer.add_summary(imgs_summary, summary_nb)
        summary_nb += 1
        print('G loss, epoch %d value %f' % (epoch, gloss))
