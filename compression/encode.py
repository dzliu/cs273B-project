import numpy as np
from keras.models import model_from_json
 
json_file = open('encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json)

f = open('labels.txt')
lines = f.readlines()[11:]
data = np.empty((len(lines), 31))
labels = np.loadtxt('clean_labels.txt')
encoded = encoder.predict(labels)
for j in range(0, len(lines)):
	l = lines[j]
	user_id = l.split('\t')[0]
	data[j, 0] = user_id
	data[j, 1:] = encoded[j]

np.savetxt('compressed.txt', data)
