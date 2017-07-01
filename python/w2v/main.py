import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

log_dir = sys.argv[1]
with open('more_life.txt', 'r', encoding='latin-1') as f:
    wfreq = dict()
    total_words = []
    for line in f:
        words = line.split(' ')
        total_words += words
        for w in words:
            if w in wfreq:
                wfreq[w] += 1
            else:
                wfreq[w] = 1
    print(len(total_words))

def build_windowset(words, wsize): #5
    wset = []
    for i in range(len(words)):
        wset += list(map(lambda e: (words[i], e), words[i+1:max(i+wsize+1, len(words))] + words[max(0, i-wsize):i]))
    return wset

def one_hot_encoding(elem, ordered_set):
    return [1 if e == elem else 0 for e in ordered_set]

vocab = list(wfreq.keys())
print(len(vocab))
idxwords = [vocab.index(e) for e in total_words]
windowset = build_windowset(idxwords, 1)

dataset = windowset
X_train, Y_train = zip(*dataset)
print('len xtrain', len(X_train))
print('X_train', X_train[0], 'Y_train', Y_train[0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

embeddings = tf.Variable(tf.random_uniform([len(vocab), 100], -1.0, 1.0), name='embeddin')

nce_weights = tf.Variable(tf.truncated_normal([len(vocab), 100], stddev=1.0/math.sqrt(100) ))
nce_biases = tf.Variable(tf.zeros([len(vocab)]))

batch_size = 128
X = tf.placeholder(tf.int32, shape=[batch_size])
Y = tf.placeholder(tf.int32, shape=[batch_size, 1])

embed = tf.nn.embedding_lookup(embeddings, X)

loss = tf.reduce_mean(tf.nn.nce_loss(
                            weights=nce_weights,
                            biases=nce_biases,
                            labels=Y,
                            inputs=embed,
                            num_sampled=64,
                            num_classes=len(vocab)
                            )
                        )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    with open(log_dir + '/metadata.tsv', 'w') as f:
        for e in vocab:
            f.write(e + '\n')
    embedding_conf.metadata_path = log_dir + '/metadata.tsv'
    projector.visualize_embeddings(writer, config)
    epoch = 0

    for i in range(0, len(X_train), batch_size):
        if epoch % 20 == 0:
            saver.save(sess, log_dir + '/model.ckpt', epoch)
        print(Y_train[i])
        _, curr_loss = sess.run([optimizer, loss], feed_dict={X:np.array(X_train[i:i+batch_size]), Y:np.array([np.array([e]) for e  in Y_train[i:i+batch_size]])})
        print(curr_loss)
        epoch += 1
