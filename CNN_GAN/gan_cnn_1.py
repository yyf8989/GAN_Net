import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 定义了一个 bn 层的参数 传入 数据 和 is_training(是否训练)
def bn(x, is_training):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training)


# 生成网络
class GNet:
    def __init__(self):
        # 定义一个变量空间
        with tf.variable_scope("gnet"):
            # 定义 w1 的值为截断正态分布
            self.w1 = tf.Variable(tf.truncated_normal(shape=[128, 1024], stddev=0.1))
            # 定义 b1 的值为 0
            self.b1 = tf.Variable(tf.zeros([1024]))

            # 定义 w2 的值为截断正态分布
            self.w2 = tf.Variable(tf.truncated_normal(shape=[1024, 32 * 7 * 7], stddev=0.1))
            # 定义 b2 的值为 0
            self.b2 = tf.Variable(tf.zeros([32 * 7 * 7]))
            # 定义conv1卷积核的 初始化值和 张量形状
            self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1, dtype=tf.float32))
            # 定义conv2卷积核的 初始化值和 张量形状
            self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1, dtype=tf.float32))

    # 定义前向计算
    def forward(self, x, is_training, reuse=False):
        # with tf.variable_scope("gnetf", reuse=tf.AUTO_REUSE):
        # 定义一个变量空间 共享 gnetf 的值
        with tf.variable_scope("gnetf", reuse=reuse):
            # leaky_relu bn 全链接第一层
            y = tf.nn.leaky_relu(bn(tf.matmul(x, self.w1) + self.b1, is_training=is_training))
            # leaky_relu bn 全链接第二层
            y = tf.nn.leaky_relu(bn(tf.matmul(y, self.w2) + self.b2, is_training=is_training))
            # 将全链接后的图片 reshape
            y = tf.reshape(y, [-1, 7, 7, 32])
            # leaky_relu bn 反卷积第一层
            deconv1 = tf.nn.leaky_relu(bn(
                tf.nn.conv2d_transpose(y, self.conv1_w, output_shape=[100, 14, 14, 16], strides=[1, 2, 2, 1],
                                       padding='SAME'), is_training=is_training))
            # leaky_relu bn 反卷积第二层
            deconv2 = tf.nn.conv2d_transpose(deconv1, self.conv2_w, output_shape=[100, 28, 28, 1], strides=[1, 2, 2, 1],
                                             padding='SAME')
            return deconv2

    # 获取变量空间中的所有变量
    def getParam(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="gnet")


# 定义判别网络
class DNet:

    def __init__(self):
        # 定义变量空间 dnet
        with tf.variable_scope("dnet"):
            # 定义卷积第一层 变量 w
            self.conv1_w = tf.Variable(
                tf.truncated_normal([3, 3, 1, 16], dtype=tf.float32, stddev=0.1))
            # 定义卷积变量 b
            self.conv1_b = tf.Variable(tf.zeros([16]))

            # 定义卷积第二层 变量 w
            self.conv2_w = tf.Variable(
                tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1))
            # 定义卷积第二层 变量 b
            self.conv2_b = tf.Variable(tf.zeros([32]))

            # 定义全链接变量 w1
            self.w1 = tf.Variable(tf.truncated_normal([7 * 7 * 32, 128], stddev=0.1))
            # 定义全链接变量 b1
            self.b1 = tf.Variable(tf.zeros([128]))

            # 定义全链接变量 w2
            self.w2 = tf.Variable(tf.truncated_normal([128, 1], stddev=0.1))

    # 定义 判别网络 前向计算
    def forward(self, x, is_training, reuse=False):
        # with tf.variable_scope("dnetf", reuse=tf.AUTO_REUSE):
        # 定义 判别网络变量空间 共享 dnetf
        with tf.variable_scope("dnetf", reuse=reuse):
            # 卷积层第一层
            conv1 = tf.nn.leaky_relu(tf.nn.conv2d(x, self.conv1_w, strides=[1, 2, 2, 1],
                                                  padding='SAME') + self.conv1_b)
            # 卷积层第二层
            conv2 = tf.nn.leaky_relu(
                bn(tf.nn.conv2d(conv1, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_b,
                   is_training=is_training))
            # 整型
            flat = tf.reshape(conv2, [-1, 7 * 7 * 32])
            # 全链接
            y = tf.matmul(flat, self.w1) + self.b1
            # 收集y的数据
            tf.summary.histogram("Discriminator pure output", y)
            # 将y batch_normal
            y = bn(y, is_training=is_training)
            # relu一下
            y0 = tf.nn.relu(y)
            # y1 = tf.nn.dropout(y0, keep_prob=0.9)
            # 输出值 matmul w2
            y_out = tf.matmul(y0, self.w2)

            # 返回y的输出值
            return y_out

    # 获取dnet变量空间中的所有值
    def getParam(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="dnet")


# 定义网络
class Net:
    def __init__(self):
        # 定义真实图片的输入值
        self.r_x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        # 定义真实图片的标签值(约束)
        self.t_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # 定义生成图片的输入值
        self.g_x = tf.placeholder(dtype=tf.float32, shape=[None, 128])
        # 定义生成图片的标签值(约束)
        self.f_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # 学习率
        self.learning_rate = 0.0002
        #
        self.beta1 = 0.5
        # 初始化生成网络
        self.gnet = GNet()
        # 初始化判别网络
        self.dnet = DNet()

        # 初始化前向计算
        self.forward()
        # 初始化后向计算
        self.backward()
        # 初始化测试
        self.test()

    # 主网络net网络的前向计算
    def forward(self):
        # 判别网络 真实图片输入的输出
        self.r_d_out = self.dnet.forward(self.r_x, is_training=True, reuse=tf.AUTO_REUSE)
        # 生成网络 生成数据输入的输出
        self.g_out = self.gnet.forward(self.g_x, is_training=True, reuse=tf.AUTO_REUSE)
        print("g_out shape: ", self.g_out.shape)
        # 判别网络 生成图片输入的输出
        self.g_d_out = self.dnet.forward(self.g_out, is_training=True, reuse=tf.AUTO_REUSE)

    def backward(self):
        # 判别网络的损失
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.r_d_out, labels=self.t_y)) \
                      + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_out, labels=self.f_y))
        # 收集 判别网络的损失
        tf.summary.scalar("Discriminator loss", self.d_loss)

        # 生成网络的损失
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_out, labels=self.t_y))
        # 收集 生成网络的损失
        tf.summary.scalar("Generator loss", self.g_loss)

        # 优化器
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # 优化 判别网络的损失
            self.d_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss,
                                                                                               var_list=self.dnet.getParam())
            # 优化生成网络的损失
            self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.g_loss,
                                                                                               var_list=self.gnet.getParam())

    # 定义测试生成网络的 情况
    def test(self):
        self.test_g_out = self.gnet.forward(self.g_x, is_training=False, reuse=tf.AUTO_REUSE)


if __name__ == '__main__':
    net = Net()
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        # 将收集的数据写入文件
        writer = tf.summary.FileWriter("./logs", sess.graph)

        # 画图
        plt.ion()
        for epoch in range(10000000):
            t_xs, _ = mnist.train.next_batch(100)
            t_xs = t_xs.reshape([100, 28, 28, 1])
            t_ys = np.ones(shape=[100, 1])

            # 随机生成 -1,1之间的值,数据形状
            f_xs = np.random.uniform(-1, 1, (100, 128))
            f_ys = np.zeros(shape=[100, 1])

            if epoch % 3 == 0:
                summary, _d_loss, _ = sess.run([merged, net.d_loss, net.d_opt],
                                               feed_dict={net.r_x: t_xs, net.t_y: t_ys, net.g_x: f_xs, net.f_y: f_ys})

            imgs, _g_loss, _ = sess.run([net.g_out, net.g_loss, net.g_opt],
                                        feed_dict={net.g_x: f_xs, net.t_y: t_ys})

            writer.add_summary(summary, epoch)

            if epoch % 100 == 0:
                print("epoch: {}, d_loss: {}, g_loss: {}".format(epoch, _d_loss, _g_loss))
                test_xs = np.random.uniform(-1, 1, (100, 128))
                test_imgs = sess.run(net.test_g_out, feed_dict={net.g_x: test_xs})
                test_img = np.reshape(test_imgs[0], (28, 28))
                img = np.reshape(imgs[0], (28, 28))

                plt.clf()
                plt.subplot(211)  # 创建字图
                plt.imshow(img)
                plt.subplot(212)
                plt.imshow(test_img)
                plt.pause(0.5)
        plt.ioff()
