import os
import numpy as np
import tqdm
import tensorflow as tf
import data_preprocessing as dp
from cnn import siamese_network
import plot_generator as pg

class SiameseNetwork:

    def __init__(self):
        self.__DATA_DIR = 'cfp-dataset/Data/Images'
        self.__TMP_DIR = 'tmp'

        self.__BATCH_SIZE = 32
        self.__ITERATIONS = 3000

        if not os.path.exists(self.__TMP_DIR):
            os.makedirs(self.__TMP_DIR)

        self.__GLOBAL_ITER = dp.global_iteration(self.__TMP_DIR + '/iteration.txt')

        print('Global iteration:', self.__GLOBAL_ITER)

        self.__train_set = []
        self.__test_set = []

        self.__shape = (105, 105, 3)

        self.__graph = tf.Graph()

        with self.__graph.as_default():
            self.__img_1 = tf.placeholder(tf.float32, shape=[None, self.__shape[0], self.__shape[1], self.__shape[2]])
            self.__img_2 = tf.placeholder(tf.float32, shape=[None, self.__shape[0], self.__shape[1], self.__shape[2]])
            self.__flags = tf.placeholder(tf.float32, shape=[None])

            self.__embeddings_1 = siamese_network(self.__img_1, reuse_variables=False)
            self.__embeddings_2 = siamese_network(self.__img_2, reuse_variables=True)

            self.__distance = tf.abs(tf.subtract(self.__embeddings_1, self.__embeddings_2))

            self.__scores = tf.contrib.layers.fully_connected(inputs=self.__distance, num_outputs=1, activation_fn=tf.nn.sigmoid,
                                                       biases_initializer=tf.truncated_normal_initializer(mean=0.5,
                                                                                                          stddev=0.01))

            self.__losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.__flags,
                                                             logits=tf.reshape(self.__scores, shape=[self.__BATCH_SIZE]))
            self.__loss = tf.reduce_mean(self.__losses)

            self.__optimizer = tf.train.AdamOptimizer(learning_rate=0.00005)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.95, use_nesterov=True)
            self.__train_op = self.__optimizer.minimize(self.__loss)

            self.__prediction = tf.cast(tf.argmax(self.__scores, axis=0), dtype=tf.int32)

            self.__saver = tf.train.Saver()

            init = tf.global_variables_initializer()

            ### SESSION ###
            self.__session = tf.Session(graph=self.__graph)

            # We must initialize all variables before we use them.
            init.run(session=self.__session)

            # reload the model if it exists and continue to train
            try:
                self.__saver.restore(self.__session, os.path.join(self.__TMP_DIR, 'model.ckpt'))
                print('Model restored')
            except:
                print('Model initialized')

    def train(self, epochs=1):
        if self.__train_set and self.__test_set:
            pass
        else:
            self.__train_set, self.__test_set = dp.load_dataset(self.__TMP_DIR, self.__DATA_DIR)

        # Open a writer to write summaries.
        self.__writer = tf.summary.FileWriter(self.__TMP_DIR, self.__session.graph)

        average_loss = 0

        for step in tqdm.tqdm(range(self.__ITERATIONS * epochs), desc='Training Siamese Network'):
            batch, label = dp.get_batch(self.__train_set, self.__BATCH_SIZE)

            pair_1 = np.array([b[0] for b in batch])
            pair_2 = np.array([b[1] for b in batch])

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            _, l = self.__session.run([self.__train_op, self.__loss], feed_dict={self.__img_1: pair_1, self.__img_2: pair_2, self.__flags: label},
                                      run_metadata=run_metadata)

            average_loss += l

            # print loss and accuracy on test set every 500 steps
            if (step % 500 == 0 and step > 0) or (step == (self.__ITERATIONS - 1)):
                correct = 0
                k = len(self.__test_set)
                for _ in range(k):
                    test, label = dp.get_one_shot_test(self.__test_set)
                    pair_1 = np.array([b[0] for b in test])
                    pair_2 = np.array([b[1] for b in test])

                    run_metadata = tf.RunMetadata()

                    pred = self.__session.run(self.__prediction, feed_dict={self.__img_1: pair_1, self.__img_2: pair_2}, run_metadata=run_metadata)
                    if pred[0] == 0:
                        correct += 1

                print('Loss:', str(average_loss / step), '\tAccuracy:', correct / k)

                with open(self.__TMP_DIR + '/log.txt', 'a', encoding='utf8') as f:
                    f.write(str(correct / k) + ' ' + str(average_loss / step) + '\n')

            if step == (self.__ITERATIONS - 1):
                self.__writer.add_run_metadata(run_metadata, 'step%d' % step, global_step=self.__GLOBAL_ITER + step + 1)

        self.__saver.save(self.__session, os.path.join(self.__TMP_DIR, 'model.ckpt'))
        dp.global_iteration(self.__TMP_DIR + '/iteration.txt', update=self.__GLOBAL_ITER + step + 1)

        pg.generate_accuracy_plot(self.__TMP_DIR + '/')
        pg.generate_loss_plot(self.__TMP_DIR + '/')

        self.__writer.close()

    def predict(self, imgs1, imgs2):
        return self.__session.run(self.__scores, feed_dict={self.__img_1: imgs1, self.__img_2: imgs2}, run_metadata=tf.RunMetadata())

    def get_embeddings(self, img):
    	return self.__session.run(self.__embeddings_1, feed_dict={self.__img_1: img}, run_metadata=tf.RunMetadata())

    def predict_with_embeddings(self, emb1, emb2):
    	return self.__session.run(self.__scores, feed_dict={self.__embeddings_1: emb1, self.__embeddings_2: emb2}, run_metadata=tf.RunMetadata())
