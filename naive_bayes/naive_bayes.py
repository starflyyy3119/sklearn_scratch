from math import *
from random import *
from csv import reader


# 参考 https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

# load a csv file
def read_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for line in csv_reader:
            dataset.append(line)
    return dataset


# convert string column to float
def str_to_float(dataset, column):
    for row in dataset:
        # 去掉前后空格
        row[column] = float(row[column].strip())


# split the data into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for i, fold in enumerate(folds):
        train = list(folds)
        train.pop(i)
        train = sum(train, [])
        test = list(fold)
        predicted = algorithm(train, test, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


# split the data by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        line = dataset[i]
        class_val = line[-1]
        if class_val not in separated:
            separated[class_val] = list()
        separated[class_val].append(line)
    return separated


# calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# calculate the mean, stdev and count for each column in the dataset
def summarize_dataset(dataset):
    x = [row[:(len(dataset[0]) - 1)] for row in dataset]
    # 解包再压包
    summarises = [(mean(column), stdev(column), len(column)) for column in zip(*x)]
    del (summarises[-1])
    return summarises


# summarize data by class
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_val, rows in separated.items():
        summaries[class_val] = summarize_dataset(rows)
    return summaries


# gaussian probability distribution function
def calculate_probability(x, mean, stdev):
    exponent = exp(-(x - mean) ** 2 / (2 * stdev ** 2))
    return 1 / sqrt(2 * pi * stdev ** 2) * exponent


# calculate class probabilities for a given new row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, col_summaries in summaries.items():
        probabilities[class_value] = col_summaries[0][2] / total_rows
        for i, col_summary in enumerate(col_summaries):
            mean, stdev, _ = col_summary
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_label = class_value
            best_prob = probability
    return best_label


def naive_bayes(train, test):
    summaries = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summaries, row)
        predictions.append(output)
    return (predictions)


seed(11)
# read iris data
dataset = read_csv("iris.csv")

# change str variable to float variable
for i in range(len(dataset[0]) - 1):
    str_to_float(dataset, i)

n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print(sum(scores) / len(scores))
