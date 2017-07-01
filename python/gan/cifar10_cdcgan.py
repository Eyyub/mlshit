import sys
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data


def get_shape(tensor):
    return tensor.get_shape().as_list()

def get_vars(modeldict):
    print(list(filter(lambda k: k.startswith('W_') or k.startswith('filters'), modeldict.keys()))) #remplacer par isinstance
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
    batch_dim = tf.shape(zs)[0]
    with tf.name_scope('G'):
        with tf.name_scope('a_z'):
            G['a_z'] = tf.matmul(zs, G_deconv_params['W_z'])
            G['bnz'] = batch_normalization(G['a_z'],  center=False, scale=False, training=is_g_training)

        with tf.name_scope('deconv1'):
            G['deconv1'] = tf.nn.conv2d_transpose(tf.reshape(tf.nn.relu(G['bnz']), [batch_dim] + G_deconv_params['reshape_z']), G_deconv_params['filters1'], [batch_dim] + G_deconv_params['deconv1_shp'], G_deconv_params['strides1'], padding='SAME')
            print('deconv', get_shape(G['deconv1']))
            G['bn1'] = batch_normalization(tf.reshape(G['deconv1'], [-1] + G_deconv_params['deconv1_shp']),  center=False, scale=False, training=is_g_training)
            G['a_1'] = tf.nn.relu(G['bn1'])

        with tf.name_scope('deconv2'):
            G['deconv2'] = tf.nn.conv2d_transpose(G['a_1'], G_deconv_params['filters2'],  [batch_dim] + G_deconv_params['deconv2_shp'], G_deconv_params['strides2'], padding='SAME')
            print('deconv2', get_shape(G['deconv2']))
            G['bn2'] = batch_normalization(tf.reshape(G['deconv2'], [-1] + G_deconv_params['deconv2_shp']),  center=False, scale=False, training=is_g_training)
            G['a_2'] = tf.nn.relu(G['bn2'])

        with tf.name_scope('deconv3'):
            G['deconv3'] = tf.nn.conv2d_transpose(G['a_2'], G_deconv_params['filters3'],  [batch_dim] + G_deconv_params['deconv3_shp'], G_deconv_params['strides3'], padding='SAME')
            print('deconv3', get_shape(G['deconv3']))
            G['a_3'] = tf.nn.tanh(tf.reshape(G['deconv3'], [-1] + G_deconv_params['deconv3_shp']))

        G['a_o'] = G['a_3']
        tf.summary.image('genimg', G['a_o'], 20)
    return G

def build_discriminator(xs, ys, D_conv_params, is_d_training, reuse=False):
    D = dict()

    with tf.variable_scope('D', reuse=reuse):
        with tf.name_scope('conv1'):
            D['conv1'] = tf.nn.conv2d(xs, D_conv_params['filters1'], D_conv_params['strides1'], padding='SAME')
            D['z_1'] = tf.concat([D['conv1'], tf.tile(tf.reshape(ys, [-1, 1, 1, get_shape(ys)[-1]]), [1, get_shape(D['conv1'])[1], get_shape(D['conv1'])[2], 1])], axis=3)
            tf.summary.histogram('z_1', D['z_1'])
            D['a_1'] = lkrelu(D['z_1'], a=0.2)

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

def random_one_hot(size):
    one_hots = np.zeros(size)
    one_hots[np.arange(size[0]), np.random.randint(size[1], size=size[0])] = 1
    return one_hots

def viz_gen_class(genimgs_tensor):
    imgs = genimgs_tensor
    with tf.name_scope('genclass'):
        summary = tf.summary.merge([tf.summary.image('c%d' % i, tf.expand_dims(tf.gather(imgs, i), 0)) for i in range(10)], collections='genclass')
    return summary

logdir = sys.argv[1]
#clean log
for filename in os.listdir(logdir):
    os.remove(os.path.join(logdir, filename))

training_set, test_set = tf.contrib.keras.datasets.cifar10.load_data()
print(training_set[1].shape)

dataset_X = 2. * (training_set[0]/ 255.) - 1.
dataset_Y = np.zeros((training_set[1].shape[0], 10))
dataset_Y[np.arange(training_set[1].shape[0]), np.reshape(training_set[1], training_set[1].shape[0])] = 1.0

batch_size = 64
zdim = 100
ydim = 10

with tf.device('/gpu:0'):

    G_deconv_params = {
        'W_z' : tf.Variable(tf.truncated_normal([zdim + ydim, 4 * 4 * 32], stddev=0.02)),
        'reshape_z' : [4, 4, 32],
        'filters1' : tf.Variable(tf.truncated_normal([5, 5, 64, 32], stddev=0.02)),
        'strides1' : [1, 2, 2, 1],
        'deconv1_shp' :  [8, 8, 64],
        'filters2' : tf.Variable(tf.truncated_normal([5, 5, 128, 64], stddev=0.02)),
        'strides2' : [1, 2, 2, 1],
        'deconv2_shp' : [16, 16, 128],
        'filters3' : tf.Variable(tf.truncated_normal([5, 5, 3, 128], stddev=0.02)),
        'strides3' : [1, 2, 2, 1],
        'deconv3_shp' : [32, 32, 3]
    }

    D_conv_params = {
        'filters1' : tf.Variable(tf.truncated_normal([5, 5, 3, 80], stddev=0.02)),
        'strides1' : [1, 2, 2, 1], # 2x2 stride
        'filters2' : tf.Variable(tf.truncated_normal([5, 5, 80 + ydim, 64], stddev=0.02)),
        'strides2' : [1, 2, 2, 1],
        'filters3' : tf.Variable(tf.truncated_normal([5, 5, 64, 32], stddev=0.02)),
        'strides3' : [1, 2, 2, 1],
        'W_o' : tf.Variable(tf.truncated_normal([4 * 4 * 32, 1], stddev=0.02))
    }

    is_g_training = tf.placeholder(tf.bool)
    is_d_training = tf.placeholder(tf.bool)

    zs = tf.placeholder(tf.float32, [None, zdim])
    g_ys = tf.placeholder(tf.float32, [None, ydim])

    xs = tf.placeholder(tf.float32, [None] + list(dataset_X.shape[1:]), name='xs')
    d_ys = tf.placeholder(tf.float32, [None, ydim])

    G = build_generator(tf.concat([zs, g_ys], axis=1), G_deconv_params, is_g_training)
    D_real = build_discriminator(xs, d_ys, D_conv_params, is_d_training)
    D_fake = build_discriminator(G['a_o'], g_ys, D_conv_params, is_d_training, reuse=True)

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

genimg_summary = viz_gen_class(G['a_o'])
total_summary = tf.summary.merge_all()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(logdir, sess.graph)
    for step_epoch in range(50000*4):
        t =  np.random.uniform(-1, 1, (batch_size, zdim))
        indices = np.random.permutation(dataset_X.shape[0])
        mini_batch_xs = dataset_X[indices[:batch_size]]
        mini_batch_ys = dataset_Y[indices[:batch_size]]
        _, dloss_curr, summary = sess.run([D_train_step, D_loss, total_summary],
                        feed_dict={xs : mini_batch_xs, d_ys : mini_batch_ys, zs : t, g_ys : mini_batch_ys,
                                    is_g_training : True, is_d_training : True})
        _, gloss_curr = sess.run([G_train_step, G_loss], feed_dict={zs : t, g_ys: random_one_hot((batch_size, 10)),
                                                                    is_g_training : True, is_d_training : True})

        if step_epoch % 500 == 1:
            imgs = sess.run(G['a_o'], feed_dict={zs : np.repeat(np.random.uniform(-1, 1, (10, zdim)), 10, axis=0), g_ys: np.tile(np.eye(10), [10, 1]), is_g_training : True, is_d_training : True})
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            fig.subplots_adjust(left=0, bottom=0,
                                   right=1, top=1, wspace=0, hspace=0.1)
            for i in range(10 * 10):
                fig.add_subplot(10, 10, i+1)
                plt.imshow((imgs[i] + 1.) / 2.)
                plt.axis('off')
            plt.savefig('images_cifar/epoch_sn_%d.jpg' % step_epoch, dpi=100)
            plt.close()

            imgs = sess.run(G['a_o'], feed_dict={zs : np.random.uniform(-1, 1, (100, zdim)), g_ys: np.repeat(np.eye(10), 10, axis=0), is_g_training : True, is_d_training : True})
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            fig.subplots_adjust(left=0, bottom=0,
                                   right=1, top=1, wspace=0, hspace=0.1)
            for i in range(10 * 10):
                fig.add_subplot(10, 10, i+1)
                plt.imshow((imgs[i] + 1.) / 2.)
                plt.axis('off')
            plt.savefig('images_cifar/epoch_%d.jpg' % step_epoch, dpi=100)
            plt.close()
        if step_epoch % 10 == 0:

            writer.add_summary(summary, step_epoch)
        print('Step %d | D loss %f | G loss %f | ' % (step_epoch, dloss_curr, gloss_curr))
