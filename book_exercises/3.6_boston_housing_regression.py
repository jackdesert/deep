import pdb
from keras.datasets import boston_housing
from keras import layers
from keras import models


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# We normalize based on the train data
# We normalize the test data too, but only using the mean and std from our train data
mean = train_data.mean(axis=0)
std  = train_data.std(axis=0)
train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    m = models.Sequential()

    # Note the variable *train_data* is a global variable because it was defined in the file
    m.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))

    m.add(layers.Dense(64, activation='relu'))
    m.add(layers.Dense(1))
    m.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return(m)



build_model()
pdb.set_trace()
