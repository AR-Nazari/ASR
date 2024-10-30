import string
import re
# import hazm
from parsivar import SpellCheck


spell_checker = SpellCheck()
# normalizer = hazm.Normalizer()
punc = string.punctuation + '،' + '؛'

def text_pipe(input_sentence):
    """
    A function to preprocess Persian text by removing punctuation,
    ignoring and protecting numbers, correcting spelling mistakes,
    and normalizing the text.

    Args:
        input_sentence (str): The input sentence in Persian.

    Returns:
        str: The processed and normalized sentence.
    """
    
    # remove punctuation
    text = ''.join([char for char in input_sentence if char not in punc])

    # ignore numbers and correct misspells
    numbers = re.findall(r'\d+', text)
    placeholders = [f"NUMBER{i}" for i in range(len(numbers))]
    for number, placeholder in zip(numbers, placeholders):
        text = text.replace(number, placeholder)
    text = spell_checker.spell_corrector(text)
    for number, placeholder in zip(numbers, placeholders):
        text = text.replace(placeholder, number)

    # normalize
    # text = normalizer.normalize(text)
    
    return text