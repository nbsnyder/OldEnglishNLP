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
data = [x for x in data if x[0] != None and x[1] != None and x[0] != "" and x[1] != ""]

# Remove unwanted characters from the spellings
data = [(Utils.removeUnwantedCharsFromWord(x[0]), Utils.removeUnwantedCharsFromWord(x[1])) for x in data]

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

partsOfSpeechArrForm = [[0] * numPartsOfSpeech] * len(partsOfSpeech)
for i in range(len(partsOfSpeech)):
    partsOfSpeechArrForm[i][partsOfSpeech[i]] = 1
partsOfSpeechArrForm = np.array(partsOfSpeechArrForm, dtype=np.uint8)

def partsOfSpeechArrFormToInt(arr):
    maxVal = np.amax(arr)
    for i in range(len(arr)):
        if np.abs(arr[i] - maxVal) < 0.000001:
            return i
    return -1

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
partsOfSpeechArrFormTrainingSet = partsOfSpeechArrForm[:trainingSetCutoff, :]

headwordsValidationSet = headwords[trainingSetCutoff:validationSetCutoff, :]
partsOfSpeechArrFormValidationSet = partsOfSpeechArrForm[trainingSetCutoff:validationSetCutoff, :]

headwordsTestingSet = headwords[validationSetCutoff:, :]
partsOfSpeechArrFormTestingSet = partsOfSpeechArrForm[validationSetCutoff:, :]
partsOfSpeechTestingSet = partsOfSpeech[validationSetCutoff:]


# Make the model
model = keras.Sequential([
    keras.layers.Dense(maxWordLength, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(numPartsOfSpeech)
])

# Compile and plot the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
)

# Fit the data
model.fit(
    headwordsTrainingSet, 
    partsOfSpeechArrFormTrainingSet, 
    validation_data=(headwordsValidationSet, partsOfSpeechArrFormValidationSet), 
    batch_size=16, 
    epochs=25
)

# Plot the model
plot_model(
    model, 
    to_file="ModelPlots/PartOfSpeechAnalysisModel.png", 
    show_shapes=True, 
    show_layer_names=True
)

# Evaluate the model
modelOutputArrForm = model.call(tf.convert_to_tensor(headwordsTestingSet), training=False).numpy()
modelOutput = [partsOfSpeechArrFormToInt(x) for x in modelOutputArrForm]

accuracy = 0
for i in range(len(headwordsTestingSet)):
    if (partsOfSpeechTestingSet[i] == modelOutput[i]):
        accuracy += 1
accuracy /= len(headwordsTestingSet)
print("\n\nAccuracy: %.3f%%" % (accuracy * 100))
