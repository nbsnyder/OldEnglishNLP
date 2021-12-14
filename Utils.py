import numpy as np

unwantedChars = set("!?,.:;\'\" []()")

charReplacements = (
    ("Ä\u0081", "ā"),
    ("Ä“", 'ē'),
    ("Ä«", "ī"),
    ("Å\u008d", "ō"),
    ("Å«", "ū"),
    ("È³", "ȳ"),
    ("Ç£", "ǣ"),
    ("Ã¦", "æ"),
    ("Ä\u2039", "ċ"),
    ("Ä¡", "ġ"),
    ("Ã¾", "þ"),
    ('Ã°', 'ð'),
    ("Ä\u2019", "Ē"),
    ("Ä\u00a0", "Ġ"),
    # Missing several less common letters
)

def replaceCharsInWord(word: str):
    for i in charReplacements:
        word = word.replace(*i)
    return word

removeUnwantedCharsFromWord = lambda word : ''.join(char for char in word if char not in unwantedChars)

wordToCharArray = lambda word, desiredWordLength : [ord(char) for char in word] + ([0] * (desiredWordLength - len(word)))

outputArrToWord = lambda charArr, maxCharValue : ''.join(chr(char) for char in (charArr * maxCharValue).astype(np.int32).tolist())
