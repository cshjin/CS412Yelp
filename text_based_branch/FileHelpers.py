__author__ = 'Natawut'

import json, SentimentFeatures
from Classes import Review


"""
This module contains functions to process the datasets -- appending tips to their
reviews, reading reviews, and generating positive/negative word files.
"""


def has_invalid_chars(text):
    for char in text:
        if ord(char) > 127 or ord(char) < 0:
            return True
    return False


def read_tips(file):
    f = open(file)
    tips = {}
    for line in f:
        tip = json.loads(line)
        text = tip["text"]
        if has_invalid_chars(text):
            continue
        tips[tip["user_id"] + tip["business_id"]] = text
    f.close()
    return tips


tips = read_tips("tips.json")


def read_reviews(file):
    global tips
    f = open(file)
    reviews = []
    counter = 1
    for line in f:
        if counter % 1000 == 0: print "Reading line " + str(counter)
        review = Review(json.loads(line))
        if has_invalid_chars(review.text):
            continue
        review_id = review.user_id + review.business_id
        if review_id in tips:
            review.text += (" " + tips[review_id])
        reviews.append(review)
        counter += 1
    return reviews

def generate_word_files(review_file, out_file_pos, out_file_neg):
    reviews = read_reviews(review_file)
    f = open(out_file_pos, "w")
    g = open(out_file_neg, "w")
    five_star_texts = [review.text for review in reviews if review.stars == 5]
    one_star_texts = [review.text for review in reviews if review.stars == 1]
    all_texts = [review.text for review in reviews]
    five_star_words = SentimentFeatures.word_counts(five_star_texts)
    one_star_words = SentimentFeatures.word_counts(one_star_texts)
    all_words = SentimentFeatures.word_counts(all_texts)
    ## Comment/uncomment depending on whether or not adjective/adverb count is used
    #SentimentFeatures.remove_top_words(all_words, 50, five_star_words, one_star_words)
    SentimentFeatures.remove_top_words(all_words, 500, five_star_words, one_star_words)
    five_star_sum = sum(five_star_words.values())
    for word in five_star_words:
        five_star_words[word] /= float(five_star_sum)
        f.write(word + " " + str(five_star_words[word]) + "\n")
    f.close()
    one_star_sum = sum(one_star_words.values())
    for word in one_star_words:
        one_star_words[word] /= -float(one_star_sum)
        g.write(word + " " + str(one_star_words[word]) + "\n")
    g.close()


def dict_from_file(file):
    f = open(file)
    d = {}
    for line in f:
        word, weight = line.strip().split()
        weight = float(weight)
        d[word] = weight
    return d


for i in range(101, 111):
    print "Processing " + str(i)
    generate_word_files("review_training_" + str(i) + ".json", "pos_words_" + str(i) + ".txt", "neg_words_" + str(i) + ".txt")
