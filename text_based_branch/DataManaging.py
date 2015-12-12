__author__ = 'Natawut'

import random
from SentimentFeatures import *


"""
This module contains functions to divide the large dataset into smaller datasets
as well as split the datasets into training and test sets.
"""

def has_invalid_chars(text):
    for char in text:
        if ord(char) > 127 or ord(char) < 0:
            return True
    return False

f = open("review_train.json")
lines = f.readlines()
f.close()
f = open("review_test.json")
for line in f.readlines():
    lines.append(line)
f.close()
lines = [line for line in lines if not has_invalid_chars(line)]

def pick_and_split(number):
    print "Processing " + str(number)
    random.shuffle(lines)
    w = open("review_training_" + str(number) + ".json", "w")
    for line in lines[:7000]: w.write(line)
    #for line in lines[:70000]: w.write(line)
    w.close()
    w = open("review_test_" + str(number) + ".json", "w")
    for line in lines[-3000:]: w.write(line)
    #for line in lines[-30000:]: w.write(line)
    w.close()

def pick_and_split_balanced(number):
    print "Processing " + str(number)
    random.shuffle(lines)
    w = open("review_training_" + str(number) + ".json", "w")
    counts = {i:0 for i in range(1,6)}
    for line in lines:
        if '"stars": 5' in line and counts[5] < 1400:
            counts[5] += 1
            w.write(line)
        if '"stars": 4' in line and counts[4] < 1400:
            counts[4] += 1
            w.write(line)
        if '"stars": 3' in line and counts[3] < 1400:
            counts[3] += 1
            w.write(line)
        if '"stars": 2' in line and counts[2] < 1400:
            counts[2] += 1
            w.write(line)
        if '"stars": 1' in line and counts[1] < 1400:
            counts[1] += 1
            w.write(line)
    w.close()
    w = open("review_test_" + str(number) + ".json", "w")
    for line in lines[-3000:]: w.write(line)
    w.close()

for i in range(101, 111):
    pick_and_split(i)