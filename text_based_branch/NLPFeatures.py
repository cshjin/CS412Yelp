__author__ = 'Natawut'

import nltk

"""
This module contains functions to extract NLP features from a review, specifically
by tagging the review and counting different POS's.
"""


def tag_text(text):
    return nltk.pos_tag(nltk.word_tokenize(text))


def adjective_count(tagged_text):
    total = 0
    for tagged_word in tagged_text:
        if tagged_word[1][0] == "J":
            total += 1
    return total


def verb_count(tagged_text):
    total = 0
    for tagged_word in tagged_text:
        if tagged_word[1][0] == "V":
            total += 1
    return total


def noun_count(tagged_text):
    total = 0
    for tagged_word in tagged_text:
        if tagged_word[1][0] == "N":
            total += 1
    return total


def adverb_count(tagged_text):
    total = 0
    for tagged_word in tagged_text:
        if tagged_word[1][0] == "R":
            total += 1
    return total