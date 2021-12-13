import mysql.connector
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

connection = mysql.connector.connect(host='localhost', database='oldengli_oea', user='root', password='password')

if not connection.is_connected():
    print("Error: could not connect to MySQL server")
    exit()

# Retrieve a list of tuples, where each tuple has two items: the spelling of a word and its headword form
cursor = connection.cursor()
cursor.execute("SELECT spelling, hw FROM words INNER JOIN entries ON entries.entry_id = words.entry_id;")
data = cursor.fetchall()

# Close the cursor and connection
cursor.close()
connection.close()

# Randomize the order of the entries
random.shuffle(data)

# Remove any entries that are missing a spelling or headword
data = [x for x in data if x[0] != None and x[1] != None]

# Make all words lowercase
data = [[x[0].lower(), x[1].lower()] for x in data]

# Remove unwanted characters from the spellings
unwantedCharacters = set("!?,.:;\'\" []")
removeUnwantedCharactersFromWord = lambda word : ''.join(char for char in word if char not in unwantedCharacters)
data = [[removeUnwantedCharactersFromWord(x[0]), x[1]] for x in data]

# Tranform words into character arrays of equal length
# Each character array's length is extended by adding 0's to the end of it
maxWordLength = max([max(len(x[0]), len(x[1])) for x in data])
wordToCharArray = lambda word : [ord(char) for char in word] + ([0] * (maxWordLength - len(word)))
data = [[wordToCharArray(x[0]), wordToCharArray(x[1])] for x in data]

# Convert data to a numpy array of type float64
data = np.array(data, dtype=np.float64)

# Normalize data so all values fall within the range [0, 1]
maxCharValue = np.amax(data)
data = data / maxCharValue

# Split data into training (80%), validation (10%), and testing (10%) sets
trainingSetCutoff = int(len(data) * 0.8)
validationSetCutoff = int(len(data) * 0.9)
spellingsTrainingSet = data[:trainingSetCutoff, 0, :]
headwordsTrainingSet = data[:trainingSetCutoff, 1, :]
spellingsValidationSet = data[trainingSetCutoff:validationSetCutoff, 0, :]
headwordsValidationSet = data[trainingSetCutoff:validationSetCutoff, 1, :]
spellingsTestingSet = data[validationSetCutoff:, 0, :]
headwordsTestingSet = data[validationSetCutoff:, 1, :]

# Make the model
inputs = keras.Input(shape=(maxWordLength))
x = layers.Dense(maxWordLength, activation="tanh")(inputs)
outputs = layers.Dense(maxWordLength, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), loss=keras.losses.CategoricalCrossentropy())

plot_model(model, to_file='ModelPlots/LemmatizationModel.png', show_shapes=True, show_layer_names=True)

# Fit the data to the model
model.fit(data[:, 0, :], data[:, 1, :], batch_size=32, epochs=10)
