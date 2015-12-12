__author__ = 'Natawut'

import re

"""
This module contains functions for cleaning and extracting textual features from reviews,
specifically word counts -- all words, capitalized words, and emphasized words.
"""

def remove_punctuation(text):
    punctuation = ['?', '.', '/', ';', ':', '!', '(', ')', ',', '[', ']', '"']
    pattern = "[" + re.escape("".join(punctuation)) + "]"
    new_text = re.sub(pattern, " ", text)
    new_text = re.sub("\'(?!(t|ll|s|d|re|m|ve)\W)", " ", new_text)
    new_text = re.sub("  +", " ", new_text)
    return new_text.strip()


def count_punctuation(text):
    punctuation = ['!', '?', '.', ';', ',', ':']
    pattern = "[" + re.escape("".join(punctuation)) + "]"
    return len(re.findall(pattern, text))


def count_cap_words(text):
    new_text = remove_punctuation(text)
    new_text = re.sub("-", "", new_text)
    words = new_text.split()
    cap_words = 0
    for word in words:
        if word.isupper() and len(word) > 2:
            cap_words += 1
    return cap_words


def count_emph_words(text):
    new_text = remove_punctuation(text).lower()
    new_text = re.sub("-", "", new_text)
    words = new_text.split()
    emph_words = 0
    for word in words:
        if re.search(r"([a-z])\1\1", word.lower()) or re.search(r"\*.+\*", word):
            emph_words += 1
    return emph_words


def word_count(text):
    new_text = remove_punctuation(text)
    return len(new_text.split())


def letter_count(text):
    return len(re.findall("[a-zA-Z]", text))