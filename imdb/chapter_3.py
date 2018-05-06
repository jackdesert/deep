import pdb
import numpy as np
from keras.datasets import imdb
from keras import layers
from keras import models
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test  = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
y_val = y_train[:10000]

partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
loss_values     = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values      = history_dict['acc']
val_acc_values  = history_dict['val_acc']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, acc_values,      'ro', label='Training Accuracy')
plt.plot(epochs, val_acc_values , 'r',  label='Validation Accuracy')
plt.plot(epochs, val_loss_values, 'b',  label='Validation loss')
plt.plot(epochs, loss_values,     'bo', label='Training loss')

plt.title('Training and Validation Loss and Accuracy')
plt.xlabel('Epochs')

plt.legend(loc='best')
plt.savefig('baseline.png')
#plt.savefig('baseline_with_one_layer.png') # Validation Accuracy about the same, but smoother between epochs
#plt.savefig('baseline_with_three_layers.png') # Validation Accuracy about the same, but noisier between epochs
#plt.savefig('baseline_with_8_hidden_units.png') # Validation Accuracy about the same, but smoother between epochs
#plt.savefig('baseline_with_32_hidden_units.png') # Validation Accuracy about the same, but noisier between epochs
#plt.savefig('baseline_with_mse.png') # Slightly worse accuracy
#plt.savefig('baseline_with_tanh.png') # Slightly worse accuracy
