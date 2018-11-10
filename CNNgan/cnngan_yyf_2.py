#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = 'YYF'
__mtime__ = '2018/11/10'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓   ┏┓
            ┏┛┻━━━┛┻┓
           ┃   ☃   ┃
           ┃ ┳┛ ┗┳ ┃
           ┃   ┻    ┃
            ┗━┓   ┏━┛
              ┃    ┗━━━┓
               ┃ 神兽保佑 ┣┓
               ┃ 永无BUG! ┏┛
                ┗┓┓┏ ━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import os
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
save_path = r"E:\Pycharmprojects\GAN_again\CNNgan\cnngan_2_ckpt\1_ckpt"


class Dnet:
    def __init__(self):
        with tf.variable_scope('D_params'):
            # 定义第一个卷积层
            self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 1, 16], dtype=tf.float32, stddev=0.1))
            self.conv1_b = tf.Variable(tf.zeros([16]))
            # 定义第二个卷积层
            self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1))
            self.conv2_b = tf.Variable(tf.zeros([32]))
            # 卷积后进行全连接输出
            self.w = tf.Variable(tf.truncated_normal([7 * 7 * 32, 128], dtype=tf.float32, stddev=0.1))
            self.b = tf.Variable(tf.zeros([128]))
            # 定义输出层w
            self.out_w = tf.Variable(tf.truncated_normal([128, 1], dtype=tf.float32, stddev=0.1))

    def forward(self, x):
        # 第一个卷积层
        conv1 = tf.nn.leaky_relu(tf.nn.conv2d(x, self.conv1_w, [1, 2, 2, 1], padding='SAME') + self.conv1_b)
        # 第二个卷积层
        # conv2_bn = tf.layers.batch_normalization(
        #     tf.nn.conv2d(conv1, self.conv2_w, [1, 2, 2, 1], padding='SAME') + self.conv2_b)
        conv2 = tf.nn.leaky_relu(tf.nn.conv2d(conv1, self.conv2_w, [1, 2, 2, 1], padding='SAME') + self.conv2_b)
        # 卷层reshape转换形状
        conv2_flat = tf.reshape(conv2, [-1, 7 * 7 * 32])
        # 第一个全连接层输入
        # mlp_bn = tf.layers.batch_normalization(tf.matmul(conv2_flat, self.w) + self.b)
        mlp = tf.nn.leaky_relu(tf.matmul(conv2_flat, self.w) + self.b)
        # 输出
        # output_bn = tf.layers.batch_normalization(tf.matmul(mlp, self.out_w))
        output = tf.nn.sigmoid(tf.matmul(mlp, self.out_w))
        # output = tf.nn.sigmoid(output_bn)

        return output

    def getparamas(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D_params')


class Gnet:
    def __init__(self):
        with tf.variable_scope('G_params'):
            # 数据输入后，然后定义第一个全连接
            self.in_w1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128, 1024], stddev=0.1))
            self.in_b1 = tf.Variable(tf.zeros([1024]))
            # 定义第二个全连接w，转换至可以输入反卷积的水平
            self.in_w2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[1024, 7 * 7 * 32], stddev=0.1))
            self.in_b2 = tf.Variable(tf.zeros([7 * 7 * 32]))
            # 定义反卷积核1的大小，和判别器的卷积核相反
            self.deconv1_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1))
            # self.conv1_b = tf.Variable(tf.zeros([32]))
            # 定义反卷积核2的大小，和判别器的卷积核相反
            self.deconv2_w = tf.Variable(tf.truncated_normal([3, 3, 1, 16], dtype=tf.float32, stddev=0.1))
            # self.conv2_b = tf.Variable(tf.zeros([16]))

    def forward(self, x):
        # 定义第一个全连接输出
        # mlp1_bn = tf.layers.batch_normalization(tf.matmul(x, self.in_w1) + self.in_b1)
        mlp1 = tf.nn.leaky_relu(tf.matmul(x, self.in_w1) + self.in_b1)
        # 定义第二个全连接输出
        # mlp2_bn = tf.layers.batch_normalization(tf.matmul(mlp1, self.in_w2) + self.in_b2)
        mlp2 = tf.nn.leaky_relu(tf.matmul(mlp1, self.in_w2) + self.in_b2)
        # 对全连接输出值进行形状转换
        mlp2_conv = tf.reshape(mlp2, [-1, 7, 7, 32])
        # 定义反卷积的第一个输出
        # deconv1_bn = tf.layers.batch_normalization(
        #     tf.nn.conv2d_transpose(mlp2_conv, self.deconv1_w, [100, 14, 14, 16], [1, 2, 2, 1], padding='SAME'))
        deconv1 = tf.nn.leaky_relu(
            tf.nn.conv2d_transpose(mlp2_conv, self.deconv1_w, [100, 14, 14, 16], [1, 2, 2, 1], padding='SAME'))
        # 定义反卷积的第二个输出
        deconv2 = tf.nn.conv2d_transpose(deconv1, self.deconv2_w, [100, 28, 28, 1], [1, 2, 2, 1], padding='SAME')

        return deconv2

    def getparams(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G_params')


class CNNgannet:
    def __init__(self):
        self.d_input_x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        # 定义判别器卷积层输入
        self.g_input_x = tf.placeholder(dtype=tf.float32, shape=[None, 128])
        # 定义生成随机样本的特征值

        self.dnet = Dnet()
        self.gnet = Gnet()

        self.forward()
        self.backward()

    def forward(self):
        self.D_out = self.dnet.forward(self.d_input_x)
        self.G_out = self.gnet.forward(self.g_input_x)
        self.G_D_out = self.dnet.forward(self.G_out)

    def backward(self):
        self.D_loss = -(tf.reduce_mean(tf.log(self.D_out)) + tf.reduce_mean(tf.log(1 - self.G_D_out)))
        # 定义损失

        self.D_opt = tf.train.GradientDescentOptimizer(0.02).minimize(self.D_loss, var_list=self.dnet.getparamas())
        # 优化器

        self.G_loss = tf.reduce_mean(-tf.log(self.G_D_out))

        # 定义损失
        self.G_opt = tf.train.GradientDescentOptimizer(0.02).minimize(self.G_loss, var_list=self.gnet.getparams())


if __name__ == '__main__':
    net = CNNgannet()
    init = tf.global_variables_initializer()
    save = tf.train.Saver()

    with tf.Session() as sess:
        # sess.run(init)
        save.restore(sess, save_path=save_path)

        plt.ion()
        i = 0
        for epoch in range(100000):
            # 训练判别器
            d_input, _ = mnist.train.next_batch(100)
            d_x = np.reshape(d_input, (100, 28, 28, 1))
            g_x = np.random.uniform(-1, 1, size=(100, 128))
            # if epoch % 3 == 0:
            #     _D_loss, _ = sess.run([net.D_loss, net.D_opt],feed_dict={net.d_input_x: d_x, net.g_input_x: g_x})
            _D_loss, _ = sess.run([net.D_loss, net.D_opt], feed_dict={net.d_input_x: d_x, net.g_input_x: g_x})

            # 训练生成器
            g_x = np.random.uniform(-1, 1, size=(100, 128))
            _G_outs, _G_loss, _ = sess.run([net.G_out, net.G_loss, net.G_opt], feed_dict={net.g_input_x: g_x})

            if epoch % 100 == 0:
                i += 1
                print('epoch:{}, D_loss:{:5.2f}, G_loss:{:5.2f}'.format(epoch, _D_loss, _G_loss))
                # 测试图片输入
                test_input_x = np.random.uniform(-1, 1, size=(100, 128))
                test_image_data = sess.run(net.G_out, feed_dict={net.g_input_x: test_input_x})
                test_image = np.reshape(test_image_data[5], (28, 28))

                # 输出上面训练后的输出图片
                train_image = np.reshape(_G_outs[5], (28, 28))

                plt.clf()
                plt.subplot(121)  # 创建字图
                plt.imshow(test_image)
                plt.title('test image')
                plt.subplot(122)
                plt.imshow(train_image)
                plt.title('train image')
                plt.pause(0.1)
                plt.savefig('E:\Pycharmprojects\GAN_again\CNNgan\cnngan_2_image\{}.png'.format(str(i).zfill(3)))
                save.save(sess, save_path=save_path)
        plt.ioff()
