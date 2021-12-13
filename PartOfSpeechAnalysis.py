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

# Retrieve a list of tuples, where each tuple has two items: a headword and its part of speech
cursor = connection.cursor()
cursor.execute("SELECT hw, part FROM entries;")
data = cursor.fetchall()

# Close the cursor and connection
cursor.close()
connection.close()

# Remove any entries that are missing a headword or part of speech
data = [x for x in data if x[0] != None and x[1] != None]

# The maximum length of a headword in this dataset
maxWordLength = 0

# Mappings between parts of speech and integers
numPartsOfSpeech = 0
partOfSpeechToInt = dict()
partsOfSpeechList = list()

for x in data:
    if x[1] not in partOfSpeechToInt:
        partOfSpeechToInt[x[1]] = numPartsOfSpeech
        partsOfSpeechList.append(x[1])
        numPartsOfSpeech += 1
    
    if len(x[0]) > maxWordLength:
        maxWordLength = len(x[0])

# Randomize the order of the data entries
random.shuffle(data)

# Change the part of speech strings into their integer representation
partsOfSpeech = [partOfSpeechToInt[x[1]] for x in data]

# Tranform headwords into character arrays of equal length
# Each character array's length is extended by adding 0's to the end of it
wordToCharArray = lambda word : [ord(char) for char in word] + ([0] * (maxWordLength - len(word)))
headwords = [wordToCharArray(x[0]) for x in data]

# Convert headwords and partsOfSpeech to numpy arrays
headwords = np.array(headwords, dtype=np.float64)
partsOfSpeech = np.array(partsOfSpeech, dtype=np.int64)

# Normalize headwords so all values fall within the range [0, 1]
maxCharValue = np.amax(headwords)
headwords = headwords / maxCharValue

# Split data into training (80%), validation (10%), and testing (10%) sets
trainingSetCutoff = int(len(data) * 0.8)
validationSetCutoff = int(len(data) * 0.9)
headwordsTrainingSet = headwords[:trainingSetCutoff, :]
partsOfSpeechTrainingSet = partsOfSpeech[:trainingSetCutoff]
headwordsValidationSet = headwords[trainingSetCutoff:validationSetCutoff, :]
partsOfSpeechValidationSet = partsOfSpeech[trainingSetCutoff:validationSetCutoff]
headwordsTestingSet = headwords[validationSetCutoff:, :]
partsOfSpeechTestingSet = partsOfSpeech[validationSetCutoff:]

# Make the model
inputs = keras.Input(shape=(maxWordLength))
x = layers.Dense(maxWordLength, activation="tanh")(inputs)
outputs = layers.Dense(1, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), loss=keras.losses.CategoricalCrossentropy())

plot_model(model, to_file='ModelPlots/PartOfSpeechAnalysisModel.png', show_shapes=True, show_layer_names=True)

# Fit the data to the model
model.fit(headwordsTrainingSet, partsOfSpeechTrainingSet, batch_size=32, epochs=10)
