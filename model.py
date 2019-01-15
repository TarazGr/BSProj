import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf
import data_preprocessing as dp
import pickle
import evaluation as eval

DATA_DIR = 'att_faces'
TMP_DIR = 'tmp'

BATCH_SIZE = 16
ITERACTIONS = 5000

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

GLOBAL_ITER = dp.global_iteraction(TMP_DIR + '/iteraction.txt')

dataset = []
category2index = {}
index = 0

for folder in os.listdir(DATA_DIR):
    directory = os.path.join(DATA_DIR, folder)
    if os.path.isdir(directory):
        #print(folder)
        if folder not in category2index:
            category2index[folder] = index
            index += 1
        d = []
        for img in os.listdir(directory):
            img = Image.open(os.path.join(directory, img)).convert('L')
            img = np.asarray(img)
            d.append(img.reshape(img.shape[0], img.shape[1], 1))
        dataset.append(d)

if not os.path.exists(TMP_DIR + '/train_set.pkl') or not os.path.exists(TMP_DIR + '/test_set.pkl'):
    train_set, test_set = dp.split_dataset(dataset, category2index, train_size=0.8)
    pickle.dump(train_set, open(TMP_DIR + '/train_set.pkl', 'wb'))
    pickle.dump(test_set, open(TMP_DIR + '/test_set.pkl', 'wb'))
else:
    print('Reloading train set and test set')
    train_set = pickle.load(open(TMP_DIR + '/train_set.pkl', 'rb'))
    test_set = pickle.load(open(TMP_DIR + '/test_set.pkl', 'rb'))
del dataset


shape = train_set[0][0].shape
#print(shape)

### MODEL ###

def model(img):

    with tf.variable_scope('convolutional_layer_1'):
        layer = tf.layers.conv2d(inputs=img, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                 bias_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0002))
        layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='same')

    with tf.variable_scope('convolutional_layer_2'):
        layer = tf.layers.conv2d(inputs=img, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                 bias_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0002))
        layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='same')

    with tf.variable_scope('convolutional_layer_3'):
        layer = tf.layers.conv2d(inputs=img, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                 bias_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0002))
        layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='same')

    with tf.variable_scope('convolutional_layer_4'):
        layer = tf.layers.conv2d(inputs=img, filters=16, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                 bias_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0002))
        layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='same')

    with tf.variable_scope('flatten_layer'):
        layer = tf.contrib.layers.flatten(layer)

    with tf.variable_scope('dense_layer_1'):
        embeddings = tf.layers.dense(inputs=layer, units=1028, activation=tf.nn.sigmoid,
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                     bias_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return embeddings


graph = tf.Graph()

with graph.as_default():
    img_1 = tf.placeholder(tf.float32, shape=[None, shape[0], shape[1], shape[2]])
    img_2 = tf.placeholder(tf.float32, shape=[None, shape[0], shape[1], shape[2]])
    flags = tf.placeholder(tf.float32, shape=[None])

    with tf.variable_scope('siamese', reuse=tf.AUTO_REUSE) as scope:
        embeddings_1 = model(img_1)
        embeddings_2 = model(img_2)

    distance = tf.abs(embeddings_1 - embeddings_2)

    scores = tf.layers.dense(inputs=distance, units=1, activation=tf.nn.sigmoid,
                             bias_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))

    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=flags, logits=tf.reshape(scores, shape=[BATCH_SIZE]))
    loss = tf.reduce_mean(losses)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.95, use_nesterov=True)
    train_op = optimizer.minimize(loss)

    prediction = tf.cast(tf.argmax(scores, axis=0), dtype=tf.int32)

    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:

    session.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(TMP_DIR, session.graph)

    # reload the model if it exists and continue to train it
    try:
        saver.restore(session, os.path.join(TMP_DIR, 'model.ckpt'))
        print('Model restored')
        print('Global epoch:', GLOBAL_ITER)
    except:
        print('Model initialized')

    average_loss = 0

    for step in tqdm.tqdm(range(ITERACTIONS), desc='Training Siamese Network'):

        batch, label = dp.get_batch(train_set, BATCH_SIZE)

        pair_1 = np.array([b[0] for b in batch])
        pair_2 = np.array([b[1] for b in batch])


        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        _, l = session.run([train_op, loss],
                           feed_dict={img_1: pair_1,
                                      img_2: pair_2,
                                      flags: label},
                           run_metadata=run_metadata)

        average_loss += l


        # print loss every 500 steps
        if (step % 500 == 0 and step > 0) or (step == (ITERACTIONS - 1)):
            correct = 0
            k = len(test_set) * len(test_set[0])
            for _ in range(k):
                test, label = dp.get_one_shot_test(test_set)
                pair_1 = np.array([b[0] for b in test])
                pair_2 = np.array([b[1] for b in test])

                run_metadata = tf.RunMetadata()

                pred = session.run(prediction,
                                   feed_dict={img_1: pair_1, img_2: pair_2}, run_metadata=run_metadata)
                if pred[0] == 0:
                    correct += 1

            print('Loss:', str(average_loss / step), '\tAccuracy:', correct / k)
        if step == (ITERACTIONS - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step, global_step=GLOBAL_ITER + step + 1)

    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))
    dp.global_iteraction(TMP_DIR + '/iteraction.txt', update=GLOBAL_ITER + step + 1)

writer.close()





