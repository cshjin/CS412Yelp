__author__ = 'Natawut'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


"""
This module contains functions for producing plots for the results.
"""


def plot_error(file, title):
    f = open(file)
    predict = []
    actual = []
    for line in f:
        p, a = map(int, line.strip().split())
        predict.append(p)
        actual.append(a)
    f.close()
    points = {(i,j): 0 for i in range(1,6) for j in range(1,6)}
    for i in range(len(predict)):
        points[(predict[i], actual[i])] += 1
    x = []
    for i in range(1,6):
        x.extend([i] * 5)
    y = range(1,6) * 5
    s = [points[(x[i],y[i])] for i in range(len(x))]
    c = ['r' if x[i] == y[i] else 'b' for i in range(len(x))]
    plt.scatter(x, y, s=s, c=c)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(title)
    plt.show()


def plot_accuracy(file1, file2):
    f = open(file1)
    g = open(file2)
    types = []
    accs1 = []
    accs2 = []
    for line in f:
        types.append(line.strip().split()[0])
        accs1.append(float(line.strip().split()[1]))
    for line in g:
        accs2.append(float(line.strip().split()[1]))
    plt.bar(range(1,7), accs1, align="edge", color="red", width=-.3)
    plt.bar(range(1,7), accs2, align="edge", color="blue", width=.3)
    red_patch = mpatches.Patch(color='red', label='Without Adj/Adv Count')
    blue_patch = mpatches.Patch(color='blue', label='With Adj/Adv Count')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xticks(range(1,7), types)
    plt.ylim([0,1])
    plt.title("Classifier Comparison by Mean Accuracy (5 Labels)")
    plt.xlabel("Classifier")
    plt.ylabel("Mean Accuracy")
    plt.show()

def plot_binaccuracy(file1, file2):
    f = open(file1)
    g = open(file2)
    types = []
    accs1 = []
    accs2 = []
    for line in f:
        types.append(line.strip().split()[0])
        accs1.append(float(line.strip().split()[2]))
    for line in g:
        accs2.append(float(line.strip().split()[2]))
    plt.bar(range(1,7), accs1, align="edge", color="red", width=-.3)
    plt.bar(range(1,7), accs2, align="edge", color="blue", width=.3)
    red_patch = mpatches.Patch(color='red', label='Without Adj/Adv Count')
    blue_patch = mpatches.Patch(color='blue', label='With Adj/Adv Count')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xticks(range(1,7), types)
    plt.ylim([0,1])
    plt.title("Classifier Comparison by Mean Accuracy (2 Labels)")
    plt.xlabel("Classifier")
    plt.ylabel("Mean Accuracy")
    plt.show()

def plot_oboaccuracy(file1, file2):
    f = open(file1)
    g = open(file2)
    types = []
    accs1 = []
    accs2 = []
    for line in f:
        types.append(line.strip().split()[0])
        accs1.append(float(line.strip().split()[3]))
    for line in g:
        accs2.append(float(line.strip().split()[3]))
    plt.bar(range(1,7), accs1, align="edge", color="red", width=-.3)
    plt.bar(range(1,7), accs2, align="edge", color="blue", width=.3)
    red_patch = mpatches.Patch(color='red', label='Without Adj/Adv Count')
    blue_patch = mpatches.Patch(color='blue', label='With Adj/Adv Count')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xticks(range(1,7), types)
    plt.ylim([0,1])
    plt.title("Classifier Comparison by Mean 'Off-by-One' Accuracy")
    plt.xlabel("Classifier")
    plt.ylabel("Mean Accuracy")
    plt.show()

def plot_exaccuracy(file1, file2):
    f = open(file1)
    g = open(file2)
    types = []
    accs1 = []
    accs2 = []
    for line in f:
        types.append(line.strip().split()[0])
        accs1.append(float(line.strip().split()[4]))
    for line in g:
        accs2.append(float(line.strip().split()[4]))
    plt.bar(range(1,7), accs1, align="edge", color="red", width=-.3)
    plt.bar(range(1,7), accs2, align="edge", color="blue", width=.3)
    red_patch = mpatches.Patch(color='red', label='Without Adj/Adv Count')
    blue_patch = mpatches.Patch(color='blue', label='With Adj/Adv Count')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xticks(range(1,7), types)
    plt.ylim([0,1])
    plt.title("Classifier Comparison by Mean Accuracy (Polarized vs. Neutral)")
    plt.xlabel("Classifier")
    plt.ylabel("Mean Accuracy")
    plt.show()

plot_accuracy("average_results_1_10.txt", "average_results_101_110.txt")
plot_binaccuracy("average_results_1_10.txt", "average_results_101_110.txt")
plot_oboaccuracy("average_results_1_10.txt", "average_results_101_110.txt")
plot_exaccuracy("average_results_1_10.txt", "average_results_101_110.txt")
