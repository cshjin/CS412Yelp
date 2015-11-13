__author__ = 'Natawut'

import json, re
from sklearn import tree, svm, neighbors
from sklearn.ensemble import RandomForestClassifier

def has_invalid_chars(text):
    for char in text:
        if ord(char) > 127:
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

def remove_punctuation(text):
    punctuation = ['?', '.', '/', ';', ':', '!', '(', ')', '-', ',']
    rx = '[' + re.escape(''.join(punctuation)) + ']'
    new_text = re.sub(rx, " ", text)
    return re.sub("  +", " ", new_text)

def add_to_dict(d, word):
    if word in d:
        d[word] += 1
    else:
        d[word] = 1

def remove_intersection(*dicts):
    if len(dicts) < 2:
        return
    intersection = set.intersection(*[set(d) for d in dicts])
    for word in intersection:
        for d in dicts:
            d.pop(word)

def remove_top_words(word_freqs, n, *dicts):
    top_words = sorted(word_freqs.keys(), key=lambda entry: word_freqs[entry], reverse=True)[:n]
    for word in top_words:
        for d in dicts:
            d.pop(word)

def remove_rare_words(threshold, *dicts):
    for d in dicts:
        rare_words = [word for word in d if d[word] <= threshold]
        for word in rare_words:
            d.pop(word)

def count_words_remove_intersection(file, tips):
    f = open(file)
    five_star_words = {}
    one_star_words = {}
    for line in f:
        review = json.loads(line)
        text = review["text"]
        if has_invalid_chars(text):
            continue
        stars = review["stars"]
        review_id = review["user_id"] + review["business_id"]
        if review_id in tips:
            text += (" " + tips[review_id])
        text = remove_punctuation(text).lower()
        words = text.split()
        for word in words:
            if stars == 5:
                add_to_dict(five_star_words, word)
            elif stars == 1:
                add_to_dict(one_star_words, word)
    f.close()
    remove_rare_words(2, five_star_words, one_star_words)
    remove_intersection(five_star_words, one_star_words)
    return five_star_words, one_star_words

def count_words_remove_frequent(file, tips):
    f = open(file)
    five_star_words = {}
    one_star_words = {}
    all_words = {}
    for line in f:
        review = json.loads(line)
        text = review["text"]
        if has_invalid_chars(text):
            continue
        stars = review["stars"]
        review_id = review["user_id"] + review["business_id"]
        if review_id in tips:
            text += (" " + tips[review_id])
        text = remove_punctuation(text).lower()
        words = text.split()
        for word in words:
            add_to_dict(all_words, word)
            if stars == 5:
                add_to_dict(five_star_words, word)
            elif stars == 1:
                add_to_dict(one_star_words, word)
    f.close()
    remove_rare_words(2, five_star_words, one_star_words)
    remove_top_words(all_words, 500, five_star_words, one_star_words)
    return five_star_words, one_star_words

def write_to_file(d, file):
    f = open(file, "w")
    words = sorted(d.keys(), key=lambda entry: d[entry], reverse=True)
    for word in words:
        f.write(word + " " + str(d[word]) + "\n")
    f.close()

def word_scores(pos_words, neg_words):
    pos_total = sum(pos_words.values())
    neg_total = sum(neg_words.values())
    all_words = {}
    for word in pos_words:
        all_words[word] = float(pos_words[word]) / pos_total
    for word in neg_words:
        score = -float(neg_words[word]) / neg_total
        if word in all_words:
            all_words[word] += score
        else:
            all_words[word] = score
    return all_words

def score_text(scores, text):
    score = 0
    for word in text.split():
        if word in scores:
            score += scores[word]
    return score

def build_from_file(file):
    f = open(file)
    words = {}
    for line in f:
        word, score = line.split()
        words[word] = int(score)
    f.close()
    return words

def score_file(file, tips, pos_word_file, neg_word_file):
    f = open(file)
    pos_words = build_from_file(pos_word_file)
    neg_words = build_from_file(neg_word_file)
    scores = word_scores(pos_words, neg_words)
    features = []
    labels = []
    for line in f:
        review = json.loads(line)
        text = review["text"]
        if has_invalid_chars(text):
            continue
        stars = review["stars"]
        review_id = review["user_id"] + review["business_id"]
        if review_id in tips:
            text += (" " + tips[review_id])
        text = remove_punctuation(text).lower()
        features.append([score_text(scores, text)])
        labels.append(stars)
    f.close()
    return features, labels

def run_count():
    tips = read_tips("yelp_academic_dataset_tip.json")
    five_star_words, one_star_words = count_words_remove_frequent("review_train.json", tips)
    write_to_file(five_star_words, "FiveStarWords.txt")
    write_to_file(one_star_words, "OneStarWords.txt")

def run_evaluation():
    tips = read_tips("yelp_academic_dataset_tip.json")
    print "Finished reading tips."
    X, Y = score_file("review_train.json", tips, "FiveStarWords.txt", "OneStarWords.txt")
    print "Finished scoring training set."
    test_X, test_Y = score_file("review_test.json", tips, "FiveStarWords.txt", "OneStarWords.txt")
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(X, Y)
    correct = sum(decision_tree.predict(test_X) == test_Y)
    print "Decision tree accuracy: " + str(correct / float(len(test_X)))
    support_vector = svm.SVC()
    support_vector = support_vector.fit(X, Y)
    correct = sum(support_vector.predict(test_X) == test_Y)
    print "SVM accuracy: " + str(correct / float(len(test_X)))
    random_forest = RandomForestClassifier(n_estimators=10)
    random_forest = random_forest.fit(X, Y)
    correct = sum(random_forest.predict(test_X) == test_Y)
    print "Random forest accuracy: " + str(correct / float(len(test_X)))
    nearest_neighbors = neighbors.KNeighborsClassifier(10)
    nearest_neighbors.fit(X, Y)
    correct = sum(nearest_neighbors.predict(test_X) == test_Y)
    print "Nearest neighbors accuracy: " + str(correct / float(len(test_X)))


run_evaluation()