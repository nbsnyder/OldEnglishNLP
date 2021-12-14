import mysql.connector
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import activations
from keras.utils.vis_utils import plot_model

import Utils

# Connect to the MySQL server
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

# Randomize the order of the data entries
random.shuffle(data)

# Split data into a list of headwords and a list of parts of speech
headwords = [x[0] for x in data]
partsOfSpeech = [x[1] for x in data]

# Replace certain character sequences in each headword
headwords = [Utils.replaceCharsInWord(x) for x in headwords]

# Find the maximum length of a headword in this dataset
maxWordLength = max([len(x) for x in headwords])

# Generate mappings between parts of speech and integers
numPartsOfSpeech = 0
partOfSpeechToInt = dict()
partsOfSpeechList = list()
for x in partsOfSpeech:
    if x not in partOfSpeechToInt:
        partOfSpeechToInt[x] = numPartsOfSpeech
        partsOfSpeechList.append(x)
        numPartsOfSpeech += 1

# Change the part of speech strings into their integer representation
partsOfSpeech = [partOfSpeechToInt[x] for x in partsOfSpeech]
partsOfSpeech = np.array(partsOfSpeech, dtype=np.uint8)

# Tranform each headword into a character array of length `maxWordLength`
# The headword is padded with \0 if the headword is not already of length `maxWordLength`
headwords = [Utils.wordToCharArray(x, maxWordLength) for x in headwords]
headwords = np.array(headwords, dtype=np.float64)

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
x = layers.Dense(maxWordLength, activation=activations.tanh)(inputs)
outputs = layers.Dense(1, activation=activations.softmax)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile and plot the model
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), loss=keras.losses.CategoricalCrossentropy())
plot_model(model, to_file="ModelPlots/PartOfSpeechAnalysisModel.png", show_shapes=True, show_layer_names=True)

# Fit the data
model.fit(headwordsTrainingSet, partsOfSpeechTrainingSet, batch_size=32, epochs=10)
