"""
Preprocess the Jester Dataset for collaborative filtering
"""
import random
import numpy as np
import pandas as pd
import simMetrics, simInstance from Similarity_metrics
import scipy.stats

# Read the dataset
def readData(filename, rowindex, columnindex):
    data_traincol = []
    data_testcol = []
    i = 0
    f = open(filename, 'rb')
    for line in f.xreadlines():
        if i in rowindex:
            ele = line.split(',')
            col_test = []
            col_train = []

            for j in range(1, len(ele)):
                rating = convertRating(ele[j])
                if j in columnindex:
                    col_test.append(rating)
                else:
                    col_train.append(rating)

            data_traincol.append(col_train)
            data_testcol.append(col_test)
        i += 1
    f.close()
    return data_traincol, data_testcol


def convertRating(raw):
    raw = float(raw)
    if raw > 90:
        rating = 0
    elif raw > 0:
        rating = 1
    else:
        rating = -1
    return rating


def findNeighbors(user, training, k):
    simMatrix = []
    for i in range(len(training)):
        sims = simInstance(simMetrics)
        sims = sims.prepareAttribute(user, training[i])
        sims = sims.computeSim()
        simMatrix = [i, sims.sim1, sims.sim2, sims.sim3]

    simDF = pd.DataFrame(simMatrix, columns = ['index', 'overlap', 'goodall', 'eskin'])
    simDF_overlap = simDF.sort(['overlap'], ascending = False)
    neighbors_overlap = simDF_overlap.iloc[0:(k+1), 0:2]

    simDF_goodall = simDF.sort(['goodall'], ascending = False)
    neighbors_goodall = simDF_goodall.iloc[0:(k+1), [0, 2]]

    simDF_eskin = simDF.sort(['eskin'], ascending = False)
    neighbors_eskin = simDF_eskin.iloc[0:(k+1), [0, 3]]

    return [neighbors_overlap, neighbors_goodall, neighbors_eskin]


def computeRecommend(user, neighbors, test):
    all_scores_bin = []
    for i in range(len(user)):
        score0 = 0
        normalization = 0
        score_bin = 0
        for n in range(len(neighbors)):
            score0 += neighbors[n][1] * test[n][i]
            normalization += neighbors[n][1]
        score = float(score0) / normalization
        if score > 0:
            score_bin = 1

        all_scores_bin.append(score_bin)
    return all_scores_bin


def predictCompare(user, scores):
    TP = 0
    FN = 0
    TN = 0
    FP = 0

    for i in range(len(user)):
        if user[i] == 1 and scores[i] == 1:
            TP += 1
        elif user[i] == 1 and scores[i] == 0:
            FN += 1
        elif user[i] == 0 and scores[i] == 0:
            TN += 1
        else:
            FP += 1
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return [precision, recall, F1]


def splitTrainTestRow(data, rowindex):
    train = []
    test = []
    for i in range(len(data)):
        if i in rowindex:
            test.append(data[i])
        else:
            train.append(data[i])
    return train, test


def experiment(train, test, test_items, k):
    all_prediction = []
    for i in range(len(test)):
        neighbors = findNeighbors(test[i], train, k)
        all_pred = []
        for j in range(len(neighbors)):
            scores = computeRecommend(test[i], neighbors[j], test_items)
            prediction = predictCompare(test[i], scores)
            all_pred.append(prediction)
        all_prediction.append(all_pred)

    return all_prediction


def computeperformance(predictions, nmetric=3):
    all_precisions = []
    all_recalls = []
    all_f1s = []
    for i in range(nmetric):
        precisions = []
        recalls = []
        f1s = []
        for j in range(len(predictions)):
            precisions.append(predictions[j][i][0])
            recalls.append(predictions[j][i][1])
            f1s.append(predictions[j][i][2])
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_f1s.append(f1s)
    return all_precisions, all_recalls, all_f1s


def resultAnalysis(results):
    avg0 = np.mean(results[0])
    avg1 = np.mean(results[1])
    avg2 = np.mean(results[2])
    pvalue01 = scipy.stats.ttest_rel(results[0], results[1])[1]
    pvalue02 = scipy.stats.ttest_rel(results[0], results[2])[1]
    pvalue12 = scipy.stats.ttest_rel(results[1], results[2])[1]

    return [avg0, avg1, avg2, pvalue01, pvalue02, pvalue12]


if __name__ == "__main__":

    filename = "jester-data-2.csv"
    #randomly select 10000 patients for the experiment
    rowindex0 = random.sample(range(0, 23500), 10000)
    #randomly select 30 items used for testing
    columnindex0 = random.sample(range(1, 101), 30)
    data_traincol, data_test_col = readData(filename, rowindex0, columnindex0)

    #randomly select 1,000 patients for testing
    rowindex1 = random.sample(range(0, 10000), 1000)

    training_row, testing_row = splitTrainTestRow(data_traincol, rowindex1)

    k = 20
    all_predictions = experiment(training_row, testing_row, data_test_col, k)

    precisions, recalls, f1s = computeperformance(all_predictions, nmetric=3)

    results_precision = resultAnalysis(precisions)
    print "Precisions:"
    print results_precision

    results_recall = resultAnalysis(recalls)
    print "Recalls:"
    print results_recall

    results_f1 = resultAnalysis(f1s)
    print "F1 Scores:"
    print results_f1