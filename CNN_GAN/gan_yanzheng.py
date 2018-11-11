import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


font = ImageFont.truetype(font="FreeMonoBold.ttf", size=25)


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    img = Image.new("RGB", (80, 30), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i in range(len(captcha_text)):
        draw.text((20 * i, 5), captcha_text[i], font=font, fill=(255, 255, 255))

    captcha_image = np.array(img)
    return captcha_text, captcha_image


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(30, 80, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (30, 80, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255. - 0.5  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 10
            return k
        k = ord(c) - 48
        if k > 9:
            raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx == 10:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


class GNet:

    def __init__(self):
        with tf.variable_scope("gnet"):
            self.w1 = tf.Variable(tf.truncated_normal(shape=[256, 512], stddev=0.1))
            self.b1 = tf.Variable(tf.zeros([512]))

            self.w2 = tf.Variable(tf.truncated_normal(shape=[512, 1024], stddev=0.1))
            self.b2 = tf.Variable(tf.zeros([1024]))

            self.w3 = tf.Variable(tf.truncated_normal(shape=[1024, IMAGE_HEIGHT * IMAGE_WIDTH], stddev=0.1))

    def forward(self, x):
        y = tf.nn.leaky_relu(tf.matmul(x, self.w1) + self.b1)
        y = tf.nn.leaky_relu(tf.matmul(y, self.w2) + self.b2)

        y = tf.matmul(y, self.w3)

        return y

    def getParam(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="gnet")


class DNet:

    def __init__(self):
        with tf.variable_scope("dnet"):
            self.conv1_w = tf.Variable(
                tf.truncated_normal([3, 3, 1, 16], dtype=tf.float32, stddev=0.1))
            self.conv1_b = tf.Variable(tf.zeros([16]))

            self.conv2_w = tf.Variable(
                tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1))
            self.conv2_b = tf.Variable(tf.zeros([32]))

            self.w1 = tf.Variable(tf.truncated_normal([8 * 20 * 32, 128], stddev=0.1))
            self.b1 = tf.Variable(tf.zeros([128]))

            self.w2 = tf.Variable(tf.truncated_normal([128, 1], stddev=0.1))

    def forward(self, x):
        conv1 = tf.nn.relu(tf.nn.conv2d(x, self.conv1_w, strides=[1, 1, 1, 1],
                                        padding='SAME') + self.conv1_b)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

        conv2 = tf.nn.relu(
            tf.nn.conv2d(pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        flat = tf.reshape(pool2, [-1, 8 * 20 * 32])

        y = tf.matmul(flat, self.w1) + self.b1
        tf.summary.histogram("Discriminator pure output", y)
        y0 = tf.nn.relu(y)
        # y1 = tf.nn.dropout(y0, keep_prob=0.9)
        y_out = tf.matmul(y0, self.w2)

        return y_out

    def getParam(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="dnet")


class Net:

    def __init__(self):
        self.r_x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        self.t_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.g_x = tf.placeholder(dtype=tf.float32, shape=[None, 256])
        self.f_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.gnet = GNet()
        self.dnet = DNet()

        self.forward()
        self.backward()

    def forward(self):
        self.r_d_out = self.dnet.forward(self.r_x)

        self.g_out = self.gnet.forward(self.g_x)
        self.g_out = tf.reshape(self.g_out, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        self.g_d_out = self.dnet.forward(self.g_out)

        # weight clipping
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.dnet.getParam()]

    def backward(self):
        # self.d_loss = tf.reduce_mean((self.r_d_out - self.t_y) ** 2) \
        #               + tf.reduce_mean((self.g_d_out - self.f_y) ** 2)
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.r_d_out, labels=self.t_y)) \
                      + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_out, labels=self.f_y))

        tf.summary.scalar("Discriminator loss", self.d_loss)
        self.d_opt = tf.train.AdamOptimizer(0.0001).minimize(self.d_loss, var_list=self.dnet.getParam())

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_out, labels=self.t_y))
        tf.summary.scalar("Generator loss", self.g_loss)
        self.g_opt = tf.train.AdamOptimizer(0.0001).minimize(self.g_loss, var_list=self.gnet.getParam())


if __name__ == '__main__':
    # Initial some necessary parameters for functions
    text, image = gen_captcha_text_and_image()
    # 图像大小
    IMAGE_HEIGHT = 30
    IMAGE_WIDTH = 80
    MAX_CAPTCHA = len(text)
    print("验证码图像channel:", image.shape)  # (30, 80, 3)
    # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
    print("验证码文本最长字符数", MAX_CAPTCHA)

    # 文本转向量
    char_set = number + ['_']  # 如果验证码长度小于4, '_'用来补齐
    CHAR_SET_LEN = len(char_set)

    # 向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每11个编码一个字符，这样顺利有，字符也有
    # vec = text2vec("5236")
    # text = vec2text(vec)
    # print(text)  # 5236
    # vec = text2vec("2468")
    # text = vec2text(vec)
    # print(text)  # 2468

    net = Net()
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./logs", sess.graph)

        plt.ion()
        for epoch in range(100000):
            t_xs, _ = get_next_batch(100)
            t_xs = t_xs.reshape([100, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
            t_ys = np.ones(shape=[100, 1])

            f_xs = np.random.uniform(-1, 1, (100, 256))
            f_ys = np.zeros(shape=[100, 1])

            if epoch % 5 == 0:
                summary, _d_loss, _, _ = sess.run([merged, net.d_loss, net.d_opt, net.clip_D],
                                                  feed_dict={net.r_x: t_xs, net.t_y: t_ys, net.g_x: f_xs,
                                                             net.f_y: f_ys})

            _g_loss, _ = sess.run([net.g_loss, net.g_opt],
                                  feed_dict={net.g_x: f_xs, net.t_y: t_ys})

            writer.add_summary(summary, epoch)

            if epoch % 100 == 0:
                print("epoch: {}, d_loss: {}, g_loss: {}".format(epoch, _d_loss, _g_loss))
                test_xs = np.random.uniform(-1, 1, (1, 256))
                imgs = sess.run(net.g_out, feed_dict={net.g_x: test_xs})
                img = np.reshape(imgs[0], [IMAGE_HEIGHT, IMAGE_WIDTH])
                plt.clf()
                plt.subplot(211)
                plt.imshow(img)
                plt.subplot(212)
                plt.imshow(t_xs[0].reshape([IMAGE_HEIGHT, IMAGE_WIDTH]))
                plt.pause(0.1)
        plt.ioff()