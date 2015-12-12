__author__ = 'Natawut'

import re

"""
This module contains functions for extracting sentiment features from reviews,
specifically using the positive/negative word lists to score a review.
"""

def remove_punctuation(text):
    punctuation = ['?', '.', '/', ';', ':', '!', '(', ')', ',', '[', ']', '"']
    pattern = "[" + re.escape("".join(punctuation)) + "]"
    new_text = re.sub(pattern, " ", text)
    new_text = re.sub("\'(?!(t|ll|s|d|re|m|ve)\W)", " ", new_text)
    new_text = re.sub("  +", " ", new_text)
    return new_text.strip()


def is_emph_word(word):
    if re.search(r"([a-z])\1\1", word.lower()):
        return True
    return False


def truncate_word(word):
    return re.sub(r"([a-z])\1\1+", r"\1\1", word.lower())


def add_to_dict(d, word):
    if word in d:
        d[word] += 1
    else:
        d[word] = 1


def remove_rare_words(d, threshold):
    rare_words = [word for word in d if d[word] <= threshold]
    for word in rare_words:
        d.pop(word)


def remove_top_words(word_freqs, n, *dicts):
    top_words = sorted(word_freqs.keys(), key=lambda entry: word_freqs[entry], reverse=True)[:n]
    for word in top_words:
        for d in dicts:
            d.pop(word)


def word_counts(texts):
    counts = {}
    for text in texts:
        for word in remove_punctuation(text).lower().split():
            add_to_dict(counts, truncate_word(word))
    remove_rare_words(counts, 2)
    return counts


def remove_intersection(*dicts):
    if len(dicts) < 2:
        return
    intersection = set.intersection(*[set(d) for d in dicts])
    for word in intersection:
        for d in dicts:
            d.pop(word)


def score_text(text, *dicts):
    score = 0
    for word in remove_punctuation(text).lower().split():
        word = truncate_word(word)
        for d in dicts:
            if word in d:
                score += d[word]
    return score