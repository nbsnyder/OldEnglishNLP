import numpy as np



def removePunctuationFromWord(word: str):
    """Removes punctuation characters from a word."""

    return ''.join(char for char in word if char not in removePunctuationFromWord.unwantedChars)

# List of punctuation characters to remove from words in the process of decoding
removePunctuationFromWord.unwantedChars = set("!?,.:;\'\" []()")



def replaceOldEnglishCharsInWord(word: str):
    """Replaces certain character sequences in a word with their original Old English character equivalents."""

    newWord = word

    for i in replaceOldEnglishCharsInWord.charReplacements:
        newWord = newWord.replace(*i)

    return newWord

# List of character replacements to make in the process of decoding a word
replaceOldEnglishCharsInWord.charReplacements = (
    ("Ä\u0081", "ā"),
    ("Ä“", "ē"),
    ("Ä«", "ī"),
    ("Å\u008d", "ō"),
    ("Å«", "ū"),
    ("È³", "ȳ"),
    ("Ç£", "ǣ"),
    ("Ã¦", "æ"),
    ("Ä\u2039", "ċ"),
    ("Ä¡", "ġ"),
    ("Ã¾", "þ"),
    ("Ã°", "ð"),
    ("Ä\u2019", "Ē"),
    ("Ä\u00a0", "Ġ"),
    # Missing several less common letters
)



def wordStringToVector(word: str, dim: int):
    """
    Returns the vector representation of a word.

    Parameters:
        word (str): The string representation of the word
        dim (int): The desired dimension of the resulting vector (dim >= len(word))

    Returns:
        vector (list): The vector of length `dim` as a list of ints
    """

    wordLength = len(word)

    if dim < wordLength:
        raise Exception("dim cannot be lower than the length of the word")

    vector = [0] * dim
    for i in range(wordLength):
        vector[i] = ord(word[i])

    return vector



def intToVector(val: int, dim: int):
    """Transform an integer into a vector of dimension `dim` by setting each value in the vector to 0 except for a value of 1 in the dimension equal to `val` (note: dimensions are 0-indexed)."""

    if val < 0:
        raise Exception("val cannot be negative")
    elif dim <= val:
        raise Exception("dim cannot be less than or equal to val (note: `dim` is the number of dimensions in the resulting array and dimensions are 0-indexed)")

    vector = [0] * dim
    vector[val] = 1

    return vector



def noramlizeWordVector(word: np.ndarray, minCharValue: int, maxCharValue: int):
    """
    Normalizes a word vector so that all values are within the range [0, 1].

    Parameters:
        word (numpy array): The unnormalized vector representation of a word
        minCharValue (int): The minimum character value in all of the data (not just this word)
        maxCharValue (int): The maximum character value in all of the data (not just this word)

    Returns:
        A numpy array (dtype=np.float64) of the normalized vector
    """

    return (word - minCharValue) / (maxCharValue - minCharValue)



def normalizedWordVectorToString(word: np.ndarray, minCharValue: int, maxCharValue: int):
    """
    Returns the string representation of a normalized word vector.

    Parameters:
        word (numpy array): The normalized vector representation of a word
        minCharValue (int): The minimum character value in all of the data (not just this word)
        maxCharValue (int): The maximum character value in all of the data (not just this word)

    Returns:
        The string representation of `word`
    """

    unNormalizedVector = (word * (maxCharValue - minCharValue)) + minCharValue
    unNormalizedVector = unNormalizedVector.astype(np.int32)
    return ''.join(chr(char) for char in unNormalizedVector)



def distanceBetweenVectors(vector1: np.ndarray, vector2: np.ndarray, minCharValue: int, maxCharValue: int):
    """
    This function takes in normalized vectors as parameters and returns the Euclidean distance between their unnormalized equivalents.

    Parameters:
        vector1 (numpy array): The normalized vector representation of a word
        vector2 (numpy array): The normalized vector representation of a word
        minCharValue (int): The minimum character value in all of the data (not just these words)
        maxCharValue (int): The maximum character value in all of the data (not just these words)

    Returns:
        The distance between the unnormalized equivalents of `vector1` and `vector2` as a float
    """

    unnormalizedVector1 = (vector1 * (maxCharValue - minCharValue)) + minCharValue
    unnormalizedVector2 = (vector2 * (maxCharValue - minCharValue)) + minCharValue
    squaredDifferences = (unnormalizedVector2 - unnormalizedVector1) ** 2

    return np.sum(squaredDifferences) ** 0.5



def partOfSpeechVectorToInt(vector: np.ndarray):
    """Converts a part-of-speech vector into its integer representation."""

    if np.size(vector) == 0:
        raise Exception("vector is empty")

    maxVal = vector[0]
    maxValIndex = 0

    for i in range(1, len(vector)):
        if vector[i] > maxVal:
            maxVal = vector[i]
            maxValIndex = i
    
    return maxValIndex
