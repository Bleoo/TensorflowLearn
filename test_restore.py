import numpy as np
import tensorflow as tf

# 定义的shape和类型要和save的一样才能加载
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

# 可以不用init
# init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'saver/save_net.ckpt')
    print("weights: ", sess.run(W))
    print("biases: ", sess.run(b))
