import sys
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data

def get_shape(tensor):
    return tensor.get_shape().as_list()

def get_vars(modeldict):
    print(list(filter(lambda k: k.startswith('W_') or k.startswith('filters'), modeldict.keys())))
    return [modeldict[v] for v in filter(lambda k: k.startswith('W_') or k.startswith('filters'), modeldict.keys())]

def lkrelu(x, a=0.01):
    return tf.maximum(a * x, x)

def batch_normalization(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn
# MNIST 28x28x1
def build_generator(zs, G_deconv_params, is_g_training):
    G = dict()

    with tf.name_scope('G'):
        with tf.name_scope('a_z'):
            G['a_z'] = tf.matmul(zs, G_deconv_params['W_z'])
            G['bnz'] = batch_normalization(G['a_z'],  center=False, scale=False, training=is_g_training)
        print('bnz', get_shape(G['bnz']), get_shape(tf.reshape(G['bnz'], G_deconv_params['reshape_z'])))
        with tf.name_scope('deconv1'):

            G['deconv1'] = tf.nn.conv2d_transpose(tf.reshape(tf.nn.relu(G['bnz']), G_deconv_params['reshape_z']), G_deconv_params['filters1'], G_deconv_params['deconv1_shp'], G_deconv_params['strides1'], padding='SAME')
            print('deconv', get_shape(G['deconv1']))
            G['bn1'] = batch_normalization(G['deconv1'],  center=False, scale=False, training=is_g_training)
            G['a_1'] = tf.nn.relu(G['bn1'])

        with tf.name_scope('deconv2'):
            G['deconv2'] = tf.nn.conv2d_transpose(G['a_1'], G_deconv_params['filters2'], G_deconv_params['deconv2_shp'], G_deconv_params['strides2'], padding='SAME')
            print('deconv2', get_shape(G['deconv2']))
            G['bn2'] = batch_normalization(G['deconv2'],  center=False, scale=False, training=is_g_training)
            G['a_2'] = tf.nn.relu(G['bn2'])

        with tf.name_scope('deconv3'):
            G['deconv3'] = tf.nn.conv2d_transpose(G['a_2'], G_deconv_params['filters3'], G_deconv_params['deconv3_shp'], G_deconv_params['strides3'], padding='SAME')
            print('deconv3', get_shape(G['deconv3']))
            G['a_3'] = tf.nn.tanh(G['deconv3'])

        G['a_o'] = G['a_3']
        tf.summary.image('genimg', tf.reshape(G['a_o'], [-1, 28, 28, 1]), 20)
    return G

def build_discriminator(xs, D_conv_params, is_d_training, reuse=False):
    D = dict()

    #with tf.name_scope('D'):
    with tf.variable_scope('D', reuse=reuse):
        with tf.name_scope('conv1'):
            D['conv1'] = tf.nn.conv2d(xs, D_conv_params['filters1'], D_conv_params['strides1'], padding='SAME')
            D['bn1'] = D['conv1']#batch_normalization(D['conv1'],  center=False, scale=False, training=is_d_training)
            tf.summary.histogram('bn', D['bn1'])
            D['a_1'] = lkrelu(D['bn1'], a=0.2)

        with tf.name_scope('conv2'):
            D['conv2'] = tf.nn.conv2d(D['a_1'], D_conv_params['filters2'], D_conv_params['strides2'], padding='SAME')
            D['bn2'] = batch_normalization(D['conv2'],  center=False, scale=False, training=is_d_training)
            D['a_2'] = lkrelu(D['bn2'], a=0.2)

        with tf.name_scope('conv3'):
            D['conv3'] = tf.nn.conv2d(D['a_2'], D_conv_params['filters3'], D_conv_params['strides3'], padding='SAME')
            D['bn3'] = batch_normalization(D['conv3'],  center=False, scale=False, training=is_d_training)
            D['a_3'] = lkrelu(D['bn3'], a=0.2)

        with tf.name_scope('o'):
            D['z_o'] = tf.matmul(tf.reshape(D['a_3'], [-1, get_shape(D_conv_params['W_o'])[0]]), D_conv_params['W_o'])
            D['a_o'] = tf.nn.sigmoid(D['z_o'])
        print('Da1', get_shape(D['a_1']))
        print('Da2', get_shape(D['a_2']))
        print('Da3', get_shape(D['a_3']))
        print('Dao', get_shape(D['a_o']))
    return D

logdir = sys.argv[1]
#clean log
for filename in os.listdir(logdir):
    os.remove(os.path.join(logdir, filename))

batch_size = 128
zdim = 100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.device('/gpu:0'):

    G_deconv_params = {
        'W_z' : tf.Variable(tf.truncated_normal([zdim, 4 * 4 * 16], stddev=0.02)),
        'reshape_z' : [-1, 4, 4, 16],
        'filters1' : tf.Variable(tf.truncated_normal([5, 5, 32, 16], stddev=0.02)),
        'strides1' : [1, 2, 2, 1],
        'deconv1_shp' :  [batch_size, 7, 7, 32],
        'filters2' : tf.Variable(tf.truncated_normal([5, 5, 64, 32], stddev=0.02)),
        'strides2' : [1, 2, 2, 1],
        'deconv2_shp' : [batch_size, 14, 14, 64],
        'filters3' : tf.Variable(tf.truncated_normal([5, 5, 1, 64], stddev=0.02)),
        'strides3' : [1, 2, 2, 1],
        'deconv3_shp' : [batch_size, 28, 28, 1]
    }

    D_conv_params = { #changer le nom
        'filters1' : tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.02)),
        'strides1' : [1, 2, 2, 1], # 2x2 stride
        'filters2' : tf.Variable(tf.truncated_normal([5, 5, 32, 16], stddev=0.02)),
        'strides2' : [1, 2, 2, 1],
        'filters3' : tf.Variable(tf.truncated_normal([5, 5, 16, 8], stddev=0.02)),
        'strides3' : [1, 2, 2, 1],
        'W_o' : tf.Variable(tf.truncated_normal([4 * 4 * 8, 1], stddev=0.02))
    }

    is_g_training = tf.placeholder(tf.bool)
    is_d_training = tf.placeholder(tf.bool)

    zs = tf.placeholder(tf.float32, [None, zdim])
    G = build_generator(zs, G_deconv_params, is_g_training)

    xs = tf.placeholder(tf.float32, [None, 784], name='xs')
    D_real = build_discriminator(tf.reshape(xs, [-1, 28, 28, 1]), D_conv_params, is_d_training)
    D_fake = build_discriminator(G['a_o'], D_conv_params, is_d_training, reuse=True)

    with tf.name_scope('Gf'):
        G_loss = -tf.reduce_mean(tf.log(D_fake['a_o']))
        tf.summary.scalar('loss', G_loss)

    with tf.name_scope('Df'):
        D_loss = -tf.reduce_mean(tf.log(D_real['a_o']) + tf.log(1.0 - D_fake['a_o']))
        tf.summary.scalar('loss', D_loss)

    G_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
    D_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
    print('G_bn', G_bn)
    print('D_bn', D_bn)

    d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
    with tf.control_dependencies(d_update_ops):
        D_train_step = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(D_loss, var_list=(get_vars(D_conv_params)))
    g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
    with tf.control_dependencies(g_update_ops):
        G_train_step = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(G_loss, var_list=(get_vars(G_deconv_params)))

total_summary = tf.summary.merge_all()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logdir, sess.graph)
    for step_epoch in range(50000):
        t =  np.random.uniform(-1, 1, (batch_size, zdim))
        mini_batch_xs, mini_batch_ys = mnist.train.next_batch(batch_size, shuffle=True)
        _, dloss_curr, summary = sess.run([D_train_step, D_loss, total_summary],
                        feed_dict={xs : 2. * mini_batch_xs - 1., zs : t,
                                    is_g_training : True, is_d_training : True})
        _, gloss_curr = sess.run([G_train_step, G_loss], feed_dict={zs : t,
                                                                    is_g_training : True, is_d_training : True})
        if step_epoch % 10 == 0:
            writer.add_summary(summary, step_epoch)
        print('Step %d | D loss %f | G loss %f | ' % (step_epoch, dloss_curr, gloss_curr))
