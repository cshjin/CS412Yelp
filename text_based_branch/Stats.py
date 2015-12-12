__author__ = 'Natawut'

import json
from Classes import Review


"""
This module contains functions for calculating statistics about the dataset and results,
specifically the different kinds of accuracy and the distribution of ratings in the dataset.
"""


def has_invalid_chars(text):
    for char in text:
        if ord(char) > 127 or ord(char) < 0:
            return True
    return False


def star_distribution(file_no):
    f = open("review_training_" + str(file_no) + ".json")
    stars = {i:0 for i in range(1,6)}
    for line in f:
        review = Review(json.loads(line))
        if has_invalid_chars(review.text):
            continue
        stars[review.stars] += 1
    f.close()
    f = open("review_test_" + str(file_no) + ".json")
    for line in f:
        review = Review(json.loads(line))
        if has_invalid_chars(review.text):
            continue
        stars[review.stars] += 1
    f.close()
    return stars


def accuracy(file_no, type):
    f = open("result_" + type + "_" + str(file_no) + ".txt")
    predict = []
    actual = []
    for line in f:
        p, a = map(int, line.strip().split())
        predict.append(p)
        actual.append(a)
    f.close()
    correct = 0
    for i in range(len(predict)):
        if predict[i] == actual[i]:
            correct += 1
    return correct/float(len(predict))


def binary_accuracy(file_no, type):
    f = open("result_" + type + "_" + str(file_no) + ".txt")
    predict = []
    actual = []
    for line in f:
        p, a = map(int, line.strip().split())
        predict.append(p)
        actual.append(a)
    f.close()
    correct = 0
    for i in range(len(predict)):
        if predict[i] == 5 and actual[i] == 5:
            correct += 1
        elif predict[i] != 5 and actual[i] != 5:
            correct += 1
    return correct/float(len(predict))

def extremes_accuracy(file_no, type):
    f = open("result_" + type + "_" + str(file_no) + ".txt")
    predict = []
    actual = []
    for line in f:
        p, a = map(int, line.strip().split())
        predict.append(p)
        actual.append(a)
    f.close()
    correct = 0
    for i in range(len(predict)):
        if predict[i] == 5 and actual[i] == 5:
            correct += 1
        elif predict[i] == 1 and actual[i] == 1:
            correct += 1
        elif predict[i] != 5 and actual[i] != 5 and predict[i] != 1 and actual[i] != 1:
            correct += 1
    return correct/float(len(predict))

def off_by_one(file_no, type):
    f = open("result_" + type + "_" + str(file_no) + ".txt")
    predict = []
    actual = []
    for line in f:
        p, a = map(int, line.strip().split())
        predict.append(p)
        actual.append(a)
    f.close()
    correct = 0
    for i in range(len(predict)):
        if abs(predict[i] - actual[i]) <= 1:
            correct += 1
    return correct/float(len(predict))


f = open("average_results_101_110.txt", "w")
types = ["ab", "dt", "rf", "5nn", "10nn", "20nn"]
for type in types:
    acc = 0
    binacc = 0
    obo = 0
    extremes = 0
    for i in range(101, 111):
        acc += accuracy(i , type)
        binacc += binary_accuracy(i, type)
        obo += off_by_one(i, type)
        extremes += extremes_accuracy(i, type)
    f.write(type + " " + str(acc / 10) + " " + str(binacc / 10) + " " + str(obo / 10) + " " + str(extremes/10) + "\n")
f.close()