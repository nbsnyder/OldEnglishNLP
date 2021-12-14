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
data = [x for x in data if x[0] != None and x[1] != None and x[0] != "" and x[1] != ""]

# The number of datapoints
numDataPoints = len(data)

# Format the data
data = [(Utils.replaceOldEnglishCharsInWord(x[0]), Utils.replaceOldEnglishCharsInWord(x[1])) for x in data]
data = [(Utils.removePunctuationFromWord(x[0]), Utils.removePunctuationFromWord(x[1])) for x in data]
data = [(x[0].lower(), x[1].lower()) for x in data]

# Randomize the order of the entries
random.shuffle(data)

# Tranform the spellings and headwords into vectors of dimension `maxWordLength`
maxWordLength = max([max(len(x[0]), len(x[1])) for x in data])
data = [(Utils.wordStringToVector(x[0], maxWordLength), Utils.wordStringToVector(x[1], maxWordLength)) for x in data]
data = np.array(data, dtype=np.float64)

# Normalize data so all values fall within the range [0, 1]
maxCharValue = np.amax(data)
minCharValue = np.amin(data)
data = [(Utils.noramlizeWordVector(x[0], minCharValue, maxCharValue), Utils.noramlizeWordVector(x[1], minCharValue, maxCharValue)) for x in data]
data = np.array(data, dtype=np.float64)


# Split data into training (80%), validation (10%), and testing (10%) sets
trainingSetCutoff = int(numDataPoints * 0.8)
validationSetCutoff = int(numDataPoints * 0.9)

spellingsTrainingSet = data[:trainingSetCutoff, 0, :]
headwordsTrainingSet = data[:trainingSetCutoff, 1, :]

spellingsValidationSet = data[trainingSetCutoff:validationSetCutoff, 0, :]
headwordsValidationSet = data[trainingSetCutoff:validationSetCutoff, 1, :]

spellingsTestingSet = data[validationSetCutoff:, 0, :]
headwordsTestingSet = data[validationSetCutoff:, 1, :]
lengthOfTestingSet = len(spellingsTestingSet)


# Make the model
model = keras.Sequential([
    keras.Input(shape=(maxWordLength)),
    keras.layers.Dense(maxWordLength * 2, activation=activations.relu),
    keras.layers.Dense(maxWordLength, activation=activations.relu)
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.Accuracy()]
)

# Make a plot of the model
plot_model(
    model,
    to_file="ModelPlots/LemmatizationModel.png",
    show_shapes=True,
    show_layer_names=True
)

# Fit the data
model.fit(
    spellingsTrainingSet,
    headwordsTrainingSet,
    validation_data=(spellingsValidationSet, headwordsValidationSet),
    batch_size=32,
    epochs=25
)

# Evaluate the model
scores = model.evaluate(spellingsTestingSet, headwordsTestingSet)
print("\nAccuracy: %.3f%%\n" % (scores[1] * 100))
modelOutput = model.call(tf.convert_to_tensor(spellingsTestingSet), training=False).numpy()


# Calculate the minimum and average distances between vectors
minDistance = 1000000000.0
minDistanceIndex = -1
averageDistance = 0.0
for i in range(lengthOfTestingSet):
    distance = Utils.distanceBetweenVectors(headwordsTestingSet[i], modelOutput[i], minCharValue, maxCharValue)
    averageDistance += distance
    if distance < minDistance:
        minDistance = distance
        minDistanceIndex = i
averageDistance = averageDistance / lengthOfTestingSet

print("Result with the minimum distance:")
print("Spelling:\t" + Utils.normalizedWordVectorToString(spellingsTestingSet[minDistanceIndex], minCharValue, maxCharValue))
print("Headword:\t" + Utils.normalizedWordVectorToString(headwordsTestingSet[minDistanceIndex], minCharValue, maxCharValue))
print("Model output:\t" + Utils.normalizedWordVectorToString(modelOutput[minDistanceIndex], minCharValue, maxCharValue))
print("Distance:\t%.3f" % (minDistance))

print("\n\nAverage distance: %.3f" % (averageDistance))
