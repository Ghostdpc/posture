import numpy as np
import model
import  tensorflow as tf
import model
import scipy.io as sio
import othermodel
HEIGHT = 128
WIDTH = 171
FRAMES = 16
CROP_SIZE = 112
CHANNELS = 3
BATCH_SIZE = 3
#reading trainning list
with open('E:/proroot/dataset/test/train_list.txt', 'r') as f:
    lines = f.read().split('\n')
tr_files = [line for line in lines if len(line) > 0]
with tf.name_scope('input_weight'):
 weights = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Define placeholders
with tf.name_scope('input_layer'):
    x_input = tf.placeholder(tf.float32, shape=(None, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), name='input_x')
    y_input = tf.placeholder(tf.int64, None, name='input_y')
training = tf.placeholder(tf.bool, name='training')

# Define the C3D model for UCF101
inputs = x_input - tf.constant([96.6], dtype=tf.float32, shape=[1, 1, 1, 1, 1])

logits = othermodel.c3d_simpler(inputs=inputs, training=training)
with tf.name_scope("input_lable"):
    labels = tf.one_hot(y_input, 7, name='labels')

# accurary opt
with tf.name_scope("accuracy"):
    correct_opt = tf.equal(tf.argmax(logits, 1), y_input, name='correct')
    acc_opt = tf.reduce_mean(tf.cast(correct_opt, tf.float32), name='accuracy')
    tf.summary.scalar("accurary", acc_opt)
# Define training opt
with tf.name_scope('loss'):  # log-损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='loss')
    tf.summary.scalar("loss", loss)
# Learning rate
global_step = tf.Variable(0, trainable=False)
with tf.name_scope('learnrate'):
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=1000,
                                               decay_rate=0.96, staircase=True)
    tf.summary.scalar("learnrate", learning_rate)
with tf.name_scope('train'):  # log-训练过程
    train_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
saver=tf.train.Saver()
with tf.Session() as sess:
 saver.restore(sess,"E:/proroot/modelsave/simper-3/")
 n_train = len(tr_files)
 for epoch in range(3):
     merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
     writer = tf.summary.FileWriter("log", tf.get_default_graph())
     tr_files = model.shuffle(tr_files)
     batch_x = np.zeros(shape=(model.BATCH_SIZE, model.FRAMES, model.CROP_SIZE, model.CROP_SIZE, model.CHANNELS), dtype=np.float32)
     batch_y = np.zeros(shape=model.BATCH_SIZE, dtype=np.float32)

     bidx = 0
     for idx, tr_file in enumerate(tr_files):
         voxel, cls = model.read_train(tr_file)

         batch_x[bidx] = voxel
         batch_y[bidx] = cls
         bidx += 1

         if (idx + 1) % model.BATCH_SIZE == 0 or (idx + 1) == n_train:
             feeds = {x_input: batch_x[:bidx], y_input: batch_y[:bidx], training: True}
             summ, _, lss, acc = sess.run([merged, train_opt, loss, acc_opt], feed_dict=feeds)
             writer.add_summary(summ, idx)

             print('%04d/%04d/%04d, loss: %.3f, acc: %.2f' % (idx / model.BATCH_SIZE, idx, n_train, lss, acc), model.ctime())
             # reset batch
             bidx = 0
     writer.close()
 saver.save(sess, "E:/proroot/ct/model.ckpt")
