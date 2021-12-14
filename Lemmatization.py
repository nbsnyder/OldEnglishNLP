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

# Retrieve a list of tuples, where each tuple has two items: the spelling of a word and its headword form
cursor = connection.cursor()
cursor.execute("SELECT spelling, hw FROM words INNER JOIN entries ON entries.entry_id = words.entry_id;")
data = cursor.fetchall()

# Close the cursor and connection
cursor.close()
connection.close()

# Remove any entries that are missing a spelling or headword
data = [x for x in data if x[0] != None and x[1] != None]

# Remove unwanted characters from the spellings
data = [(Utils.removeUnwantedCharsFromWord(x[0]), Utils.removeUnwantedCharsFromWord(x[1])) for x in data]

# Replace certain character sequences in each word
data = [(Utils.replaceCharsInWord(x[0]), Utils.replaceCharsInWord(x[1])) for x in data]

# Make all words lowercase
data = [(x[0].lower(), x[1].lower()) for x in data]

# Randomize the order of the entries
random.shuffle(data)

# Tranform each word into a character array of length `maxWordLength`
# The word is padded with \0 if the word is not already of length `maxWordLength`
maxWordLength = max([max(len(x[0]), len(x[1])) for x in data])
data = [(Utils.wordToCharArray(x[0], maxWordLength), Utils.wordToCharArray(x[1], maxWordLength)) for x in data]

# Convert data to a numpy array of type float64
data = np.array(data, dtype=np.float64)

# Normalize data so all values fall within the range [0, 1]
maxCharValue = np.amax(data)
data /= maxCharValue

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
x = layers.Dense(maxWordLength, activations.relu)(inputs)
x = layers.Dense(maxWordLength * 2, activations.relu)(x)
outputs = layers.Dense(maxWordLength, activations.relu)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile and plot the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.Accuracy()])
plot_model(model, to_file="ModelPlots/LemmatizationModel.png", show_shapes=True, show_layer_names=True)

# Fit the data
model.fit(spellingsTrainingSet, headwordsTrainingSet, validation_data=(spellingsValidationSet, headwordsValidationSet), batch_size=32, epochs=25, verbose=1)

# Evaluate the model
scores = model.evaluate(spellingsTestingSet, headwordsTestingSet, verbose=1)
print("\nAccuracy: %.3f%%\n" % (scores[1] * 100))
modelOutput = model.call(tf.convert_to_tensor(spellingsTestingSet), training=False).numpy()

# Calculate the minimum and average distances between vectors
minDistance = 1000000000
minDistanceIndex = -1
averageDistance = 0
for i in range(len(spellingsTestingSet)):
    # Calculate Euclidean distance between the two vectors
    distance = np.sum(((headwordsTestingSet[i] - modelOutput[i]) * maxCharValue) ** 2) ** 0.5
    averageDistance += distance
    if distance < minDistance:
        minDistance = distance
        minDistanceIndex = i
averageDistance = averageDistance / len(spellingsTestingSet)

print("Result with the minimum distance:")
print("Spelling:\t" + Utils.outputArrToWord(spellingsTestingSet[minDistanceIndex], maxCharValue))
print("Headword:\t" + Utils.outputArrToWord(headwordsTestingSet[minDistanceIndex], maxCharValue))
print("Model output:\t" + Utils.outputArrToWord(modelOutput[minDistanceIndex], maxCharValue))
print("Distance:\t%.3f" % (minDistance))

print("\n\nAverage distance: %.3f" % (averageDistance))
