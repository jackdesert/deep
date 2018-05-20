# BOSTON HOUSING
# (MY INTERPRETATION)

# I decided I will learn more if I study the concept, then recreate the
# experiment from what I learned.

import pdb
import pathlib

from keras.datasets import boston_housing
from keras import models
from keras import layers

import numpy as np

import matplotlib
matplotlib.use('Agg') # Using Agg because it can run without an X-server
import matplotlib.pyplot as plt


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalize the data along Axis 0 so keras has a better starting point
# Use mean and std based on trainging set (not test set) for both
train_mean = np.mean(train_data, axis=0)
train_std  = np.std(train_data, axis=0)

train_data -= train_mean
train_data /= train_std

test_data -= train_mean
test_data /= train_std

units = 64
k = 4
epochs = 500
all_mae_histories = []

def build_model():
    model = models.Sequential() # Sequential means a linear stack of layers



    # How many hidden units should be in this model??
    # Why do we use relu as opposed to some other activation (or none at all)?
    # What is the final layer?


    # Each layer needs an "activation"
    model.add(layers.Dense(units, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(units, activation='relu'))

    # Final layer is (1) for scalar (output) regression
    model.add(layers.Dense(1))

    # Compile
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return(model)



def b_split(v_index):

    # chunk length
    clen = len(train_data) // k

    v_start = v_index * clen
    v_end   = v_start + clen
    validation_data    = train_data[v_start : v_end]
    validation_targets = train_targets[v_start : v_end]

    train_data_before = train_data[: v_start]
    train_data_after  = train_data[v_end :]
    train_data_both   = np.concatenate((train_data_before, train_data_after))

    train_targets_before = train_targets[: v_start]
    train_targets_after  = train_targets[v_end :]
    train_targets_both   = np.concatenate((train_targets_before, train_targets_after))

    return((train_data_both, train_targets_both), (validation_data, validation_targets))


for i in range(k):
    (train_data_part, train_targets_part), (validation_data, validation_targets) = b_split(i)
    m = build_model()
    history = m.fit(train_data_part,
                    train_targets_part,
                    epochs=epochs,
                    verbose=1,
                    batch_size=1,
                    validation_data=(validation_data, validation_targets))
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)



for color, hist in zip(['bo', 'ro', 'yo', 'ko'], all_mae_histories): plt.plot(hist, color)

mean = np.mean(all_mae_histories, axis=0)
plt.plot(mean, 'm')

minimum = np.min(mean)
plt.plot((0, epochs), (minimum, minimum), 'm-')

plt.title(f'Validation with {units} units. (min={round(minimum, 3)})')

directory = '/tmp/plots'
pathlib.Path(directory).mkdir(exist_ok=True)

filename = f'{directory}/3.6-validation-with-{units}-units-and-{epochs}-epochs.png'
plt.savefig(filename)

print(f'saved to {filename}')































