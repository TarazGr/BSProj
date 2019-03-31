import os
from PIL import Image
import numpy as np
import tqdm
import tensorflow as tf
import data_preprocessing as dp
import pickle
import plot_generator as pg

DATA_DIR = 'cfp-dataset/Data/Images'
TMP_DIR = 'tmp'

BATCH_SIZE = 32
ITERACTIONS = 3000

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

GLOBAL_ITER = dp.global_iteraction(TMP_DIR + '/iteraction.txt')

print('Global iteraction:', GLOBAL_ITER)


if not os.path.exists(TMP_DIR + '/train_set.pkl') or not os.path.exists(TMP_DIR + '/test_set.pkl'):
    print('Loading dataset...')
    dataset = []
    category2index = {}
    index = 0
    for folder in os.listdir(DATA_DIR):
        if folder not in category2index:
            category2index[folder] = index
            index += 1
        imgs = []
        directory = os.path.join(DATA_DIR, folder)
        dir = os.path.join(directory, 'frontal')
        # print(folder)
        for img in os.listdir(dir):
            img = Image.open(os.path.join(dir, img)).convert('RGB').resize((105, 105), Image.LANCZOS)
            img = np.asarray(img)
            imgs.append(img)
        dataset.append(imgs)
    #print(len(dataset))
    #print(category2index)
    train_set, test_set = dp.split_dataset(dataset, category2index, train_size=0.75)
    del dataset
    pickle.dump(train_set, open(TMP_DIR + '/train_set.pkl', 'wb'))
    pickle.dump(test_set, open(TMP_DIR + '/test_set.pkl', 'wb'))
else:
    print('Reloading train set and test set...')
    train_set = pickle.load(open(TMP_DIR + '/train_set.pkl', 'rb'))
    test_set = pickle.load(open(TMP_DIR + '/test_set.pkl', 'rb'))


#print(len(train_set))
#print(len(test_set))

shape = train_set[0][0].shape

#print(shape)

# MODEL #

def siamese_network(img, reuse_variables=False):

    with tf.name_scope('siamese'):

        with tf.variable_scope('conv1') as scope:
            layer = tf.contrib.layers.conv2d(inputs=img, num_outputs=32, kernel_size=[10, 10], padding='VALID', activation_fn=tf.nn.relu,
                                             biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01), scope=scope,
                                             reuse=reuse_variables)
            layer = tf.contrib.layers.max_pool2d(layer, kernel_size=[2, 2], stride=2, padding='VALID')

        with tf.variable_scope('conv2') as scope:
            layer = tf.contrib.layers.conv2d(inputs=layer, num_outputs=64, kernel_size=[7, 7], padding='VALID', activation_fn=tf.nn.relu,
                                             biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01), scope=scope,
                                             reuse=reuse_variables)
            layer = tf.contrib.layers.max_pool2d(layer, kernel_size=[2, 2], stride=2, padding='VALID')

        with tf.variable_scope('conv3') as scope:
            layer = tf.contrib.layers.conv2d(inputs=layer, num_outputs=64, kernel_size=[4, 4], padding='VALID', activation_fn=tf.nn.relu,
                                             biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01), scope=scope,
                                             reuse=reuse_variables)
            layer = tf.contrib.layers.max_pool2d(layer, kernel_size=[2, 2], stride=2, padding='VALID')

        with tf.variable_scope('conv4') as scope:
            layer = tf.contrib.layers.conv2d(inputs=layer, num_outputs=128, kernel_size=[4, 4], padding='VALID', activation_fn=tf.nn.relu,
                                             biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01), scope=scope,
                                             reuse=reuse_variables)
            layer = tf.contrib.layers.max_pool2d(layer, kernel_size=[2, 2], stride=2, padding='VALID')

        with tf.variable_scope('flatten') as scope:
            layer = tf.contrib.layers.flatten(inputs=layer)

        with tf.variable_scope('fc') as scope:
            layer = tf.contrib.layers.fully_connected(inputs=layer, num_outputs=4096, activation_fn=tf.nn.sigmoid,
                                                      biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01),
                                                      scope=scope, reuse=reuse_variables)

        return layer


graph = tf.Graph()

with graph.as_default():
    img_1 = tf.placeholder(tf.float32, shape=[None, shape[0], shape[1], shape[2]])
    img_2 = tf.placeholder(tf.float32, shape=[None, shape[0], shape[1], shape[2]])
    flags = tf.placeholder(tf.float32, shape=[None])

    embeddings_1 = siamese_network(img_1, reuse_variables=False)
    embeddings_2 = siamese_network(img_2, reuse_variables=True)

    distance = tf.abs(tf.subtract(embeddings_1, embeddings_2))

    scores = tf.contrib.layers.fully_connected(inputs=distance, num_outputs=1, activation_fn=tf.nn.sigmoid,
                                               biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))

    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=flags, logits=tf.reshape(scores, shape=[BATCH_SIZE]))
    loss = tf.reduce_mean(losses)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.00005)
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
    except:
        print('Model initialized')

    average_loss = 0

    for step in tqdm.tqdm(range(ITERACTIONS), desc='Training Siamese Network'):
        batch, label = dp.get_batch(train_set, BATCH_SIZE)

        pair_1 = np.array([b[0] for b in batch])
        pair_2 = np.array([b[1] for b in batch])

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        _, l = session.run([train_op, loss], feed_dict={img_1: pair_1, img_2: pair_2, flags: label},
                           run_metadata=run_metadata)

        average_loss += l

        # print loss and accuracy on test set every 500 steps
        if (step % 500 == 0 and step > 0) or (step == (ITERACTIONS - 1)):
            correct = 0
            k = len(test_set) * len(test_set[0])
            for _ in range(k):
                test, label = dp.get_one_shot_test(test_set)
                pair_1 = np.array([b[0] for b in test])
                pair_2 = np.array([b[1] for b in test])

                run_metadata = tf.RunMetadata()

                pred = session.run(prediction, feed_dict={img_1: pair_1, img_2: pair_2}, run_metadata=run_metadata)
                if pred[0] == 0:
                    correct += 1

            print('Loss:', str(average_loss / step), '\tAccuracy:', correct / k)

            with open(TMP_DIR + '/log.txt', 'a', encoding='utf8') as f:
                f.write(str(correct / k) + ' ' + str(average_loss / step) + '\n')

        if step == (ITERACTIONS - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step, global_step=GLOBAL_ITER + step + 1)

    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))
    dp.global_iteraction(TMP_DIR + '/iteraction.txt', update=GLOBAL_ITER + step + 1)

pg.generate_accuracy_plot()
pg.generate_loss_plot()

writer.close()
