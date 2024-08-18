import string
import hazm
from parsivar import SpellCheck


spell_checker = SpellCheck()
normalizer = hazm.Normalizer()
punc = string.punctuation + '،' + '؛'

def text_pipe(input_sentence):
    text = ''.join([char for char in input_sentence if char not in punc])
    text = spell_checker.spell_corrector(text)
    text = normalizer.normalize(text)
    return text