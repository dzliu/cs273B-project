import numpy as np

from keras.layers import Input, Dense
from keras.models import Model


# this is the size of our encoded representations
encoding_dim = 10  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_dim = 734

# this is our input placeholder
input_data = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(256, activation='relu')(input_data)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
# this model maps an input to its reconstruction
autoencoder = Model(input=input_data, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

data = np.loadtxt('clean_labels.txt')
train = data[:7000,:]
test = data[7000:,:]
train = train.astype('float32') / 255.
test = test.astype('float32') / 255.
autoencoder.fit(train, train,
                nb_epoch=30,
                batch_size=256,
                shuffle=True,
                validation_data=(test, test))

