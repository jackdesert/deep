import pdb
from keras.datasets import reuters
from keras import layers
from keras import models
import numpy as np
import readline

class Runner:
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode(self, sample_index):
        # This works on the data BEFORE one-hot encoding.
        # It still has the correct order
        sample = self.train_data[sample_index]
        o = ' '.join([self.reverse_word_index.get(i - 3, '?') for i in sample])
        return(o)

    def label_hist(self):
        h = np.histogram(self.train_labels, range(46))
        dd = { index: count for index, count in zip(h[1], h[0])}
        return(dd)

    def vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results

    def to_one_hot(self, labels, dimension=46):
        results = np.zeros((len(labels), dimension))
        for i, label in enumerate(labels):
            results[i, label] = 1
        return(results)

    def run(self):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(46, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        vectorized_train_data = self.vectorize_sequences(self.train_data)
        vectorized_test_data  = self.vectorize_sequences(self.test_data)

        one_hot_train_labels = self.to_one_hot(self.train_labels)
        one_hot_test_labels  = self.to_one_hot(self.test_labels)

        main_train_data       = vectorized_train_data[1000:]
        validation_train_data = vectorized_train_data[:1000]

        main_train_labels       = one_hot_train_labels[1000:]
        validation_train_labels = one_hot_train_labels[:1000]

        history = model.fit(main_train_data,
                            main_train_labels,
                            epochs=9,
                            batch_size=512,
                            validation_data=(validation_train_data, validation_train_labels))
        predictions = model.predict(validation_train_data)
        pdb.set_trace()



rr = Runner()
rr.run()





