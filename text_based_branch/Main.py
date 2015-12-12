__author__ = 'Natawut'

import SentimentFeatures, TextFeatures, NLPFeatures, DateFeatures
from FileHelpers import *
from sklearn import tree, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif


"""
This module contains the main functions for training and testing the classifiers. After the
feature vectors are built, each classifier is trained and then tested on the corresponding
test set. Results are output to a file, either as the raw accuracy or as a listing of classifier
predictions vs actual labels. The latter is more useful for efficient analysis.
"""


def build_feature_vector(reviews, pos_words, neg_words):
    counter = 1
    for review in reviews:
        if counter % 1000 == 0: print "Processing review " + str(counter)
        counter += 1
        """
        Feature 0: sentiment score
        """
        sent_score = SentimentFeatures.score_text(review.text, pos_words, neg_words)
        review.features.append(sent_score)

        """
        Feature 1: punctuation count
        """
        punct_count = TextFeatures.count_punctuation(review.text)
        review.features.append(punct_count)

        """
        Feature 2: cap word count
        """
        cap_count = TextFeatures.count_cap_words(review.text)
        review.features.append(cap_count)

        """
        Feature 3: emphasized word count
        """
        emph_count = TextFeatures.count_emph_words(review.text)
        review.features.append(emph_count)

        """
        Feature 4: word count
        """
        word_count = TextFeatures.word_count(review.text)
        review.features.append(word_count)

        """
        Feature 5: letter count

        letter_count = TextFeatures.letter_count(review.text)
        review.features.append(letter_count)
        """
        """
        Feature 6: adjective count
        """
        #print "Tagging..."
        tagged_text = NLPFeatures.tag_text(review.text)
        #print "Counting adjectives"
        adj_count = NLPFeatures.adjective_count(tagged_text)
        review.features.append(adj_count)

        """
        Feature 7: adverb count
        """
        #print "Counting adverbs"
        adv_count = NLPFeatures.adverb_count(tagged_text)
        review.features.append(adv_count)

        '''
        Feature 8: noun count

        noun_count = NLPFeatures.noun_count(tagged_text)
        review.features.append(noun_count)
        """
        """
        Feature 9: verb count

        verb_count = NLPFeatures.verb_count(tagged_text)
        review.features.append(verb_count)
        """
        '''
        """
        Feature 10: weekday/weekend
        """
        weekday = DateFeatures.is_weekday(review)
        review.features.append(weekday)

def run(file_no):
    result = open("results_" + str(file_no) + ".txt", "w")
    reviews_training = read_reviews("review_training_" + str(file_no) + ".json")
    reviews_test = read_reviews("review_test_" + str(file_no) + ".json")
    print "Finished reading reviews"
    pos_words = dict_from_file("pos_words_" + str(file_no) + ".txt")
    neg_words = dict_from_file("neg_words_" + str(file_no) + ".txt")
    print "Finished reading pos/neg words"
    build_feature_vector(reviews_training, pos_words, neg_words)
    build_feature_vector(reviews_test, pos_words, neg_words)
    print "Finished building feature vectors"
    X_training = [review.features for review in reviews_training]
    Y_training = [review.stars for review in reviews_training]
    X_test = [review.features for review in reviews_test]
    Y_test = [review.stars for review in reviews_test]

    print "Fitting classifiers"
    random_forest = RandomForestClassifier(n_estimators=20)
    random_forest = random_forest.fit(X_training, Y_training)
    correct = sum(random_forest.predict(X_test) == Y_test)
    result.write("Random forest accuracy: " + str(correct / float(len(X_test))) + "\n")
    result.write("RF feature importance: " + str(random_forest.feature_importances_) + "\n")
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(X_training, Y_training)
    correct = sum(decision_tree.predict(X_test) == Y_test)
    result.write("Decision tree accuracy: " + str(correct / float(len(X_test))) + "\n")
    ada_boost = AdaBoostClassifier()
    ada_boost = ada_boost.fit(X_training, Y_training)
    correct = sum(ada_boost.predict(X_test) == Y_test)
    result.write("AdaBoost accuracy: " + str(correct / float(len(X_test))) + "\n")
    nearest_neighbors = neighbors.KNeighborsClassifier(5)
    nearest_neighbors = nearest_neighbors.fit(X_training, Y_training)
    correct = sum(nearest_neighbors.predict(X_test) == Y_test)
    result.write("5 nearest neighbors accuracy: " + str(correct / float(len(X_test))) + "\n")
    nearest_neighbors = neighbors.KNeighborsClassifier(10)
    nearest_neighbors = nearest_neighbors.fit(X_training, Y_training)
    correct = sum(nearest_neighbors.predict(X_test) == Y_test)
    result.write("10 nearest neighbors accuracy: " + str(correct / float(len(X_test))) + "\n")
    nearest_neighbors = neighbors.KNeighborsClassifier(20)
    nearest_neighbors = nearest_neighbors.fit(X_training, Y_training)
    correct = sum(nearest_neighbors.predict(X_test) == Y_test)
    result.write("20 nearest neighbors accuracy: " + str(correct / float(len(X_test))) + "\n")
    result.close()
    print "Finished with " + str(file_no)

def find_error(file_no):

    reviews_training = read_reviews("review_training_" + str(file_no) + ".json")
    reviews_test = read_reviews("review_test_" + str(file_no) + ".json")
    print "Finished reading reviews"
    pos_words = dict_from_file("pos_words_" + str(file_no) + ".txt")
    neg_words = dict_from_file("neg_words_" + str(file_no) + ".txt")
    print "Finished reading pos/neg words"
    build_feature_vector(reviews_training, pos_words, neg_words)
    build_feature_vector(reviews_test, pos_words, neg_words)
    print "Finished building feature vectors"
    X_training = [review.features for review in reviews_training]
    Y_training = [review.stars for review in reviews_training]
    X_test = [review.features for review in reviews_test]
    Y_test = [review.stars for review in reviews_test]

    print "Fitting classifiers"
    result = open("result_rf_" + str(file_no) + ".txt", "w")
    random_forest = RandomForestClassifier(n_estimators=20)
    random_forest = random_forest.fit(X_training, Y_training)
    prediction = random_forest.predict(X_test)
    for i in range(len(prediction)):
        result.write(str(prediction[i]) + " " + str(Y_test[i]) + "\n")
    result.close()

    result = open("result_dt_" + str(file_no) + ".txt", "w")
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(X_training, Y_training)
    prediction = decision_tree.predict(X_test)
    for i in range(len(prediction)):
        result.write(str(prediction[i]) + " " + str(Y_test[i]) + "\n")
    result.close()

    result = open("result_ab_" + str(file_no) + ".txt", "w")
    ada_boost = AdaBoostClassifier()
    ada_boost = ada_boost.fit(X_training, Y_training)
    prediction = ada_boost.predict(X_test)
    for i in range(len(prediction)):
        result.write(str(prediction[i]) + " " + str(Y_test[i]) + "\n")
    result.close()

    result = open("result_5nn_" + str(file_no) + ".txt", "w")
    nearest_neighbors = neighbors.KNeighborsClassifier(5)
    nearest_neighbors = nearest_neighbors.fit(X_training, Y_training)
    prediction = nearest_neighbors.predict(X_test)
    for i in range(len(prediction)):
        result.write(str(prediction[i]) + " " + str(Y_test[i]) + "\n")
    result.close()

    result = open("result_10nn_" + str(file_no) + ".txt", "w")
    nearest_neighbors = neighbors.KNeighborsClassifier(10)
    nearest_neighbors = nearest_neighbors.fit(X_training, Y_training)
    prediction = nearest_neighbors.predict(X_test)
    for i in range(len(prediction)):
        result.write(str(prediction[i]) + " " + str(Y_test[i]) + "\n")
    result.close()

    result = open("result_20nn_" + str(file_no) + ".txt", "w")
    nearest_neighbors = neighbors.KNeighborsClassifier(20)
    nearest_neighbors = nearest_neighbors.fit(X_training, Y_training)
    prediction = nearest_neighbors.predict(X_test)
    for i in range(len(prediction)):
        result.write(str(prediction[i]) + " " + str(Y_test[i]) + "\n")
    result.close()
    print "Finished with " + str(file_no)

for i in range(101, 111):
    find_error(i)