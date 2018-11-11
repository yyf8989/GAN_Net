import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class DNet:

    def __init__(self):
        with tf.variable_scope("D_PARAM"):
            self.in_w = tf.Variable(tf.truncated_normal(shape=[784, 512], stddev=0.01))
            self.in_b = tf.Variable(tf.zeros([512]))

            self.in_w2 = tf.Variable(tf.truncated_normal(shape=[512, 512], stddev=0.01))
            self.in_b2 = tf.Variable(tf.zeros([512]))

            self.out_w = tf.Variable(tf.truncated_normal(shape=[512, 1], stddev=0.01))

    def forward(self, x):
        y = tf.nn.relu(tf.matmul(x, self.in_w) + self.in_b)
        y = tf.nn.relu(tf.matmul(y, self.in_w2) + self.in_b2)
        return tf.sigmoid(tf.matmul(y, self.out_w))

    def getParams(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="D_PARAM")


class GNet:
    def __init__(self):
        with tf.variable_scope("G_PARAM"):
            self.in_w = tf.Variable(tf.truncated_normal(shape=[128, 512], stddev=0.01))
            self.in_b = tf.Variable(tf.zeros([512]))

            self.in_w2 = tf.Variable(tf.truncated_normal(shape=[512, 512], stddev=0.01))
            self.in_b2 = tf.Variable(tf.zeros([512]))

            self.out_w = tf.Variable(tf.truncated_normal(shape=[512, 784], stddev=0.01))

    def forward(self, x):
        y = tf.nn.leaky_relu(tf.matmul(x, self.in_w) + self.in_b)
        y = tf.nn.leaky_relu(tf.matmul(y, self.in_w2) + self.in_b2)
        return tf.matmul(y, self.out_w)

    def getParams(self):

        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="G_PARAM")


class Net:

    def __init__(self):
        self.r_x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.g_x = tf.placeholder(dtype=tf.float32, shape=[None, 128])

        self.gnet = GNet()
        self.dnet = DNet()

        self.forward()
        self.backward()

    def forward(self):
        self.out_r_y = self.dnet.forward(self.r_x)
        self.out_g_y = self.gnet.forward(self.g_x)
        self.out_f_y = self.dnet.forward(self.out_g_y)

    def backward(self):
        self.d_r_loss = tf.reduce_mean(tf.log(self.out_r_y))
        self.d_g_loss = tf.reduce_mean(tf.log(1 - self.out_f_y))
        self.d_loss = -(self.d_r_loss + self.d_g_loss)

        self.d_opt = tf.train.AdamOptimizer(0.0001).minimize(self.d_loss, var_list=self.dnet.getParams())

        self.g_loss = tf.reduce_mean(-tf.log(self.out_f_y))

        self.g_opt = tf.train.AdamOptimizer(0.0001).minimize(self.g_loss, var_list=self.gnet.getParams())


if __name__ == '__main__':
    net = Net()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(1000000):
            r_xs, _ = mnist.train.next_batch(1000)
            g_xs = np.random.normal(scale=0.01,size=[100, 128])
            _d_loss, _g_loss,_= sess.run([net.d_loss, net.g_loss, net.d_opt], feed_dict={net.r_x: r_xs, net.g_x: g_xs})

            g_xs = np.random.normal(scale=0.01,size=[100, 128])
            sess.run([net.g_opt], feed_dict={net.g_x: g_xs})

            plt.ion()
            if epoch % 100 == 0:
                print(_d_loss, _g_loss)

                test_g_xs = np.random.normal(scale=0.1,size=[1, 128])
                test_img_data = sess.run([net.out_g_y], feed_dict={net.g_x: test_g_xs})
                test_img = np.reshape(test_img_data, (28, 28))
                plt.imshow(test_img)
                plt.pause(0.1)
