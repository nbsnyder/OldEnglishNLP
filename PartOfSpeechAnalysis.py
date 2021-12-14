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

# The number of datapoints
numDataPoints = len(data)

# Randomize the order of the data entries
random.shuffle(data)

# Split data into a list of headwords and a list of parts of speech
headwords_str = [x[0] for x in data]
partsOfSpeech_str = [x[1] for x in data]

# Format the headwords and parts of speech
headwords_str = [Utils.removePunctuationFromWord(x) for x in headwords_str]
headwords_str = [Utils.replaceOldEnglishCharsInWord(x) for x in headwords_str]
partsOfSpeech_str = [Utils.removePunctuationFromWord(x) for x in partsOfSpeech_str]

# Find the maximum length of a headword in this dataset
maxWordLength = max([len(x) for x in headwords_str])

# Generate mappings between parts of speech and integers
numPartsOfSpeech = 0
partOfSpeechToInt = dict()
partsOfSpeechList = list()
for x in partsOfSpeech_str:
    if x not in partOfSpeechToInt:
        partOfSpeechToInt[x] = numPartsOfSpeech
        partsOfSpeechList.append(x)
        numPartsOfSpeech += 1

# Change the part of speech string representations into their integer representations
partsOfSpeech_int = [partOfSpeechToInt[x] for x in partsOfSpeech_str]
partsOfSpeech_int = np.array(partsOfSpeech_int, dtype=np.uint8)

# Change the part of speech integer representations into their vector representations
partsOfSpeech_vector = [Utils.intToVector(x, numPartsOfSpeech) for x in partsOfSpeech_int]
partsOfSpeech_vector = np.array(partsOfSpeech_vector, dtype=np.uint8)

# Tranform each headword into a vector of dimension `maxWordLength`
headwords_vector = [Utils.wordStringToVector(x, maxWordLength) for x in headwords_str]
headwords_vector = np.array(headwords_vector, dtype=np.float64)

# Normalize headword vectors so all values fall within the range [0, 1]
maxCharValue = np.amax(headwords_vector)
minCharValue = np.amin(headwords_vector)
headwords_vector = [Utils.noramlizeWordVector(word, minCharValue, maxCharValue) for word in headwords_vector]
headwords_vector= np.array(headwords_vector, dtype=np.float64)

# Split data into training (80%), validation (10%), and testing (10%) sets
trainingSetCutoff = int(numDataPoints * 0.8)
validationSetCutoff = int(numDataPoints * 0.9)

headwordsTrainingSet = headwords_vector[:trainingSetCutoff, :]
partsOfSpeechTrainingSet = partsOfSpeech_vector[:trainingSetCutoff, :]

headwordsValidationSet = headwords_vector[trainingSetCutoff:validationSetCutoff, :]
partsOfSpeechValidationSet = partsOfSpeech_vector[trainingSetCutoff:validationSetCutoff, :]

headwordsTestingSet = headwords_vector[validationSetCutoff:, :]
partsOfSpeechTestingSet_vector = partsOfSpeech_vector[validationSetCutoff:, :]
partsOfSpeechTestingSet_int = partsOfSpeech_int[validationSetCutoff:]
lengthOfTestingSet = len(headwordsTestingSet)


# Make the model
model = keras.Sequential([
    keras.Input(shape=(maxWordLength)),
    keras.layers.Dense(maxWordLength * 2, activation=activations.relu),
    keras.layers.Dense(maxWordLength, activation=activations.relu),
    keras.layers.Dense(numPartsOfSpeech, activation=activations.relu)
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
)

# Make a plot of the model
plot_model(
    model,
    to_file="ModelPlots/PartOfSpeechAnalysisModel.png",
    show_shapes=True,
    show_layer_names=True
)

# Fit the data
model.fit(
    headwordsTrainingSet,
    partsOfSpeechTrainingSet,
    validation_data=(headwordsValidationSet, partsOfSpeechValidationSet),
    batch_size=16,
    epochs=25
)


# Evaluate the model
modelOutput_vector = model.call(tf.convert_to_tensor(headwordsTestingSet), training=False).numpy()
modelOutput_int = [Utils.partOfSpeechVectorToInt(x) for x in modelOutput_vector]

accuracy = 0
for i in range(lengthOfTestingSet):
    if (partsOfSpeechTestingSet_int[i] == modelOutput_int[i]):
        accuracy += 1
accuracy /= lengthOfTestingSet
print("\n\nAccuracy: %.3f%%" % (accuracy * 100))
