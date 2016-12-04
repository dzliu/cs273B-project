import numpy as np

from keras.layers import Input, Dense
from keras.models import Model


# this is the size of our encoded representations
encoding_dim = 30 
input_dim = 734

# this is our input placeholder
input_data = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(512, activation='relu')(input_data)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input=input_data, output=decoded)
autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')

encoder = Model(input=input_data, output=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_output = autoencoder.layers[5](encoded_input)
decoder_output = autoencoder.layers[6](decoder_output)
decoder_output = autoencoder.layers[7](decoder_output)
decoder = Model(input=encoded_input, output=decoder_output)
from keras.callbacks import TensorBoard

data = np.loadtxt('clean_labels.txt')
train = data[:8000,:]
test = data[8000:,:]
autoencoder.fit(train, train,
                nb_epoch=250,
                batch_size=256,
                shuffle=True,
                validation_data=(test, test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# serialize model to JSON
def serialize(model, filename):
	model_json = model.to_json()
	with open(filename, "w") as json_file:
	    json_file.write(model_json)

serialize(encoder, "encoder.json")
serialize(decoder, "decoder.json")
serialize(autoencoder, "autoencoder.json")
