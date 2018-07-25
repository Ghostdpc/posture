from dataload import  *
from sklearn.utils import shuffle
from time import ctime
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#sdasdasdasd
HEIGHT = 128
WIDTH = 171
FRAMES = 16
CROP_SIZE = 112
CHANNELS = 3
BATCH_SIZE = 3






def c3d_ucf101_finetune(inputs, training, weights=None):
    """
    C3D network for ucf101 dataset fine-tuned from weights pretrained on Sports1M
    :param inputs: Tensor inputs (batch, depth=16, height=112, width=112, channels=3), should be means subtracted
    :param training: A boolean tensor for training mode (True) or testing mode (False)
    :param weights: pretrained weights, if None, return network with random initialization
    :return: Output tensor for 101 classes
    """

    # create c3d network with pretrained Sports1M weights
    net = tf.layers.conv3d(inputs=inputs, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[0]),
                           bias_initializer=tf.constant_initializer(weights[1]),name="conv3d1")
    tf.summary.histogram("conv3d1-weight1",weights[0])
    tf.summary.histogram("conv3d1-weight2", weights[1])
    net = tf.layers.average_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME',name="averagepooling1")
    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[2]),
                           bias_initializer=tf.constant_initializer(weights[3]),name="conv3d2")
    tf.summary.histogram("conv3d2-weight1",weights[2])
    tf.summary.histogram("conv3d2-weight2", weights[3])
    net = tf.layers.average_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME',name="averagepooling2")
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[4]),
                           bias_initializer=tf.constant_initializer(weights[5]),name="conv3d3")
    tf.summary.histogram("conv3d3-weight1",weights[4])
    tf.summary.histogram("conv3d3-weight2", weights[5])
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[6]),
                           bias_initializer=tf.constant_initializer(weights[7]),name="conv3d4")
    tf.summary.histogram("conv3d4-weight1",weights[6])
    tf.summary.histogram("conv3d4-weight2", weights[7])
    net = tf.layers.average_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME',name="averagepooling3")
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[8]),
                           bias_initializer=tf.constant_initializer(weights[9]),name="conv3d5")
    tf.summary.histogram("conv3d5-weight1",weights[8])
    tf.summary.histogram("conv3d5-weight2", weights[9])
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[10]),
                           bias_initializer=tf.constant_initializer(weights[11]),name="conv3d6")
    tf.summary.histogram("conv3d6-weight1",weights[10])
    tf.summary.histogram("conv3d6-weight2", weights[11])
    net = tf.layers.average_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME',name="averagepooling4")
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],name="pad1")

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[12]),
                           bias_initializer=tf.constant_initializer(weights[13]),name="conv3d7")
    tf.summary.histogram("conv3d7-weight1",weights[12])
    tf.summary.histogram("conv3d7-weight2", weights[13])
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],name="pad2")
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[14]),
                           bias_initializer=tf.constant_initializer(weights[15]),name="conv3d8")
    tf.summary.histogram("conv3d8-weight1",weights[14])
    tf.summary.histogram("conv3d8-weight2", weights[15])
    net = tf.layers.average_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME',name="averagepooling5")

    net = tf.layers.flatten(net,name="flatten1")
    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[16]),
                          bias_initializer=tf.constant_initializer(weights[17]),name="dense1")
    net = tf.identity(net, name='fc1')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training,name="dropout1")

    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[18]),
                          bias_initializer=tf.constant_initializer(weights[19]),name="dense2")
    net = tf.identity(net, name='fc2')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training,name="dropout2")

    net = tf.layers.dense(inputs=net, units=7, activation=tf.nn.softmax,
                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
                          bias_initializer=tf.zeros_initializer(),name="dense3")

    net = tf.identity(net, name='logits')

    return net


def read_train(tr_file):
    path, frm, cls = tr_file.split(' ')
    start = np.random.randint(int(frm) - FRAMES)

    frame_dir = 'E:/proroot/dataset/traindata/hmdb/'

    v_paths = [frame_dir + path +'/'+ '%d.jpg' % (f + 1) for f in range(start, start + FRAMES)]

    offsets = randcrop(scales=[128, 112, 96, 84], size=(128, 171))
    voxel = clipread(v_paths, offsets, size=(128, 171), crop_size=(112, 112), mode='RGB')

    is_flip = np.random.rand(1, 1).squeeze() > 0.5
    if is_flip:
        voxel = np.flip(voxel, axis=2)

    return voxel, np.float32(cls)





def demo_finetune():
    # Demo of training on UCF101
    with open('E:/proroot/dataset/test/train_list.txt', 'r') as f:
        lines = f.read().split('\n')
    tr_files = [line for line in lines if len(line) > 0]
    with tf.name_scope('input_weight'):
     weights = sio.loadmat('E:/proroot/dataset/pretrain/c3d_ucf101_tf.mat', squeeze_me=True)['weights']

    # Define placeholders
    with tf.name_scope('input_layer'):
     x_input = tf.placeholder(tf.float32, shape=(None, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), name='input_x')
     y_input = tf.placeholder(tf.int64, None, name='input_y')
    training = tf.placeholder(tf.bool, name='training')

    # Define the C3D model for UCF101
    inputs = x_input - tf.constant([96.6], dtype=tf.float32, shape=[1, 1, 1, 1, 1])

    logits = c3d_ucf101_finetune(inputs=inputs, training=training,weights=weights)
    with tf.name_scope("input_lable"):
     labels = tf.one_hot(y_input,7, name='labels')

    # Some operations
    with tf.name_scope("accuracy"):
     correct_opt = tf.equal(tf.argmax(logits, 1), y_input, name='correct')
     acc_opt = tf.reduce_mean(tf.cast(correct_opt, tf.float32), name='accuracy')
     tf.summary.scalar("accurary", acc_opt)
    # Define training opt
    with tf.name_scope('loss'):  # 损失
     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='loss')
     tf.summary.scalar("loss", loss)
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('learnrate'):
     learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=1000,
                                                decay_rate=0.96, staircase=True)
     tf.summary.scalar("learnrate",learning_rate)
    with tf.name_scope('train'):  # 训练过程
     train_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

       # 将训练日志写入到logs文件夹下
        n_train = len(tr_files)
        for epoch in range(3):
            merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
            writer = tf.summary.FileWriter("log", tf.get_default_graph())
            tr_files = shuffle(tr_files)
            batch_x = np.zeros(shape=(BATCH_SIZE, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), dtype=np.float32)
            batch_y = np.zeros(shape=BATCH_SIZE, dtype=np.float32)

            bidx = 0
            for idx, tr_file in enumerate(tr_files):
                voxel, cls = read_train(tr_file)

                batch_x[bidx] = voxel
                batch_y[bidx] = cls
                bidx += 1

                if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_train:
                    feeds = {x_input: batch_x[:bidx], y_input: batch_y[:bidx], training: True}
                    summ,_, lss, acc = sess.run([merged,train_opt, loss, acc_opt], feed_dict=feeds)
                    writer.add_summary(summ,idx)
                   # writer = tf.summary.FileWriter('logs', sess.graph)
                    #writer.close()

                    print('%04d/%04d/%04d, loss: %.3f, acc: %.2f' % (idx / BATCH_SIZE, idx, n_train, lss, acc), ctime())
                    # reset batch
                    bidx = 0

            writer.close()
        saver.save(sess,"E:/proroot/projfilemodel/")


if __name__ == '__main__':

     demo_finetune()

