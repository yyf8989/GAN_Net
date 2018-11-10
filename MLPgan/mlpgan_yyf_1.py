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
save_path = r"E:\Pycharmprojects\GAN_again\MLPgan\mlp_ganckpt\1_ckpt"


class Dnet:
    # 定义判别器为三个全连接组成
    def __init__(self):
        with tf.variable_scope('D_params'):
            # 定义第1个mlp全连接层
            self.in_w1 = tf.Variable(tf.truncated_normal(shape=[784, 256], stddev=0.01))
            self.in_b1 = tf.Variable(tf.zeros([256]))

            # 定义第2个mlp全连接层
            self.in_w2 = tf.Variable(tf.truncated_normal(shape=[256, 512], stddev=0.01))
            self.in_b2 = tf.Variable(tf.zeros([512]))

            # 定义第3个mlp全连接层
            self.out_w = tf.Variable(tf.truncated_normal(shape=[512, 1], stddev=0.01))
            # self.out_b = tf.Variable(tf.zeros([1024])) 因为是输出层，不要偏值

    def forward(self, x):
        # 生成第1个全连接结果
        mlp1 = tf.nn.leaky_relu(tf.matmul(x, self.in_w1) + self.in_b1)
        # 生成第2个全连接结果
        mlp2 = tf.nn.leaky_relu(tf.matmul(mlp1, self.in_w2) + self.in_b2)
        # 生成第3个全连接结果,使用sigmoid激活
        out_put = tf.nn.sigmoid(tf.matmul(mlp2, self.out_w))

        return out_put

    def getparams(self):
        # 收取所有参数值
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D_params')


class Gnet:
    # 定义全连接生成判别网络，基本结构清晰，然后进行形状输出
    def __init__(self):
        with tf.variable_scope('G_params'):
            # 定义第1个mlp全连接层
            self.in_w1 = tf.Variable(tf.truncated_normal(shape=[128, 512], stddev=0.01))
            self.in_b1 = tf.Variable(tf.zeros([512]))

            # 定义第2个mlp全连接层
            self.in_w2 = tf.Variable(tf.truncated_normal(shape=[512, 512], stddev=0.01))
            self.in_b2 = tf.Variable(tf.zeros([512]))

            # 定义第3个mlp全连接层
            self.out_w = tf.Variable(tf.truncated_normal(shape=[512, 784], stddev=0.01))
            # self.out_b = tf.Variable(tf.zeros([1024])) 因为是输出层，不要偏值

    def forward(self, x):
        # 生成第1个全连接结果
        mlp1 = tf.nn.sigmoid(tf.matmul(x, self.in_w1) + self.in_b1)
        # 生成第2个全连接结果
        mlp2 = tf.nn.sigmoid(tf.matmul(mlp1, self.in_w2) + self.in_b2)
        # 生成第3个全连接结果,因为要保持
        out_put = tf.matmul(mlp2, self.out_w)

        return out_put

    def getparams(self):
        # 收取所有的参数
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G_params')


class GanNet:
    def __init__(self):
        self.input_x = tf.placeholder(tf.float32, shape=[None, 784])
        # 定义输入形状为[None(代表未知，后面赋值), 784]
        self.input_false = tf.placeholder(tf.float32, shape=[None, 128])
        # 随机给出128个特征，后续生成假数据样本

        self.dnet = Dnet()
        # 实例化判别网络
        self.gnet = Gnet()
        # 实例化生成网络

        self.forward()
        # 内部调用前向运算
        self.backward()
        # 内部调用后向运算

    def forward(self):
        self.D_output = self.dnet.forward(self.input_x)
        # 输出由输入mnist图片产生的结果
        self.G_output = self.gnet.forward(self.input_false)
        # 输出由生成数据产生的结果
        self.G_D_output = self.dnet.forward(self.G_output)
        # 将生成器产生的数据输入判别器生成结果

    def backward(self):
        self.D_loss = -(tf.reduce_mean(tf.log(self.D_output)) + tf.reduce_mean(tf.log(1 - self.G_D_output)))
        # self.D_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_output, labels=tf.ones_like(self.D_output)))
        # self.G_D_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G_D_output, labels=tf.zeros_like(self.G_D_output)))
        # self.D_loss = self.D_loss + self.G_D_loss
        # 定义判别网络的损失，由定义公式而来
        self.D_opt = tf.train.GradientDescentOptimizer(0.01).minimize(self.D_loss, var_list=self.dnet.getparams())
        # 进行判别模型的优化，使用随机梯度下降优化器

        self.G_loss = tf.reduce_mean(-tf.log(self.G_D_output))
        # self.G_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G_output, labels=tf.ones_like(self.G_output)))
        # 定义生成网络的损失，由定义公式而来
        self.G_opt = tf.train.GradientDescentOptimizer(0.01).minimize(self.G_loss, var_list=self.gnet.getparams())


if __name__ == '__main__':
    net = GanNet()
    init = tf.global_variables_initializer()
    save = tf.train.Saver()
    i = 1
    with tf.Session() as sess:
        # sess.run(init)
        save.restore(sess, save_path=save_path)

        for epoch in range(1000000):

            input_xs, _ = mnist.train.next_batch(100)
            # 取用mnist数据集中的图片
            input_false_xs = np.random.normal(scale=0.1, size=(100, 128))
            # 生成128个随机特征值矩阵
            _D_loss, _G_loss, _ = sess.run([net.D_loss, net.G_loss, net.D_opt],
                                           feed_dict={net.input_x: input_xs, net.input_false: input_false_xs})
            # sess run D_loss,G_loss,D_OPT,然后接收随时的返回值

            # 训练生成网络
            input_falsex = np.random.normal(scale=0.1, size=(100, 128))
            sess.run(net.G_opt, feed_dict={net.input_false: input_falsex})

            plt.ion()
            # 打开连续画图

            if epoch % 1000 == 0:
                print('epoch:{}, D_loss:{:5.2f}, G_loss:{:5.2f}'.format(epoch, _D_loss, _G_loss))
                # 打印出对应的损失

                test_data = np.random.normal(scale=0.1, size=(1, 128))
                # 生成测试数据
                test_img_data = sess.run([net.G_output], feed_dict={net.input_false: test_data})
                test_img = np.reshape(test_img_data, (28, 28))
                plt.imshow(test_img)
                plt.pause(0.1)
                # 画图
                plt.savefig('E:\Pycharmprojects\GAN_again\MLPgan\Images\{}.png'.format(str(i).zfill(3)))
                # 保存图
                i += 1
                save.save(sess, save_path=save_path)
                # 保存checkpoint

            plt.ioff()
            # 关闭连续画图
