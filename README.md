# Old English Natural Language Processing

This is my exploration into Natural Language Processing in Old English. There has been very little work done in this area, so I wanted to try it out using basic Artificial Neural Network (ANN) methods. I wrote scripts for the lemmatization and part-of-speech tagging of Old English words and tested how effective these scripts were on the Old English words found in classic Old English stories.

Both scripts use data from the database of [Old English Aerobics](http://www.oldenglishaerobics.net/). The lemmatization script trains a neural network to transform the words as they are seen in these stories into their lemma/headword forms. The part of speech analysis script trains a neural network to determine the parts of speech of the words in the stories.

All of my code is in the Python files Lemmatization.py, PartOfSpeechAnalysis.py, and Utils.py, and my research paper on the exploration is "An Exploration into Natural Language Processing in Old English.pdf".
