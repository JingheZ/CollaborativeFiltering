"""
Preprocess the Jester Dataset for collaborative filtering;
Collaborative filtering for online joke recommendation 
"""
import random
import numpy as np
import pandas as pd
import Similarity_metrics
import scipy.stats
import collections

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
        rating = 0
    return rating


def varDistribution(train):
    train1 = np.array(train)
    train2 = train1.transpose()
    train3 = list(train2)
    p_dict = []
    for i in range(len(train3)):
        n1 = list(train3[i]).count(1)
        p1 = float(n1) / len(train3[i])
        p_dict.append({0: 1-p1, 1: p1})

    return p_dict



def findNeighbors(user, training, k, p_dict):
    simMatrix = []
    for i in range(len(training)):
        sims = Similarity_metrics.prepareAttribute(user, training[i], p_dict)
        weighted_sims = Similarity_metrics.computeSim(sims)
        weighted_sims.insert(0, training[i][0])
        simMatrix.append(weighted_sims)

    simDF = pd.DataFrame(simMatrix, columns = ['index', 'overlap', 'goodall', 'eskin'])
    simDF_overlap = simDF.sort_values('overlap', ascending = False)
    neighbors_overlap = simDF_overlap.iloc[0:(k+1), 0:2]

    simDF_goodall = simDF.sort_values('goodall', ascending = False)
    neighbors_goodall = simDF_goodall.iloc[0:(k+1), [0, 2]]

    simDF_eskin = simDF.sort_values('eskin', ascending = False)
    neighbors_eskin = simDF_eskin.iloc[0:(k+1), [0, 3]]

    return [neighbors_overlap, neighbors_goodall, neighbors_eskin]


def computeRecommend(user, neighbors, test, threshold):
    all_scores_bin = []
    for i in range(len(user)):
        score0 = 0
        normalization = 0
        score = 0
        score_bin = 0
        for n in range(len(neighbors)):
            neighbors1 = neighbors.values.tolist()
            neighbor_index = int(neighbors1[n][0])
            score0 += neighbors1[n][1] * test[neighbor_index][i]
            normalization += neighbors1[n][1]
        if normalization > 0:
            score = float(score0) / normalization
        if score > threshold:
            score_bin = 1

        all_scores_bin.append(score_bin)
    return all_scores_bin


def predictCompare(user, scores):
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    precision = 0
    recall = 0
    F1 = 0
    for i in range(len(user)):
        if user[i] == 1 and scores[i] == 1:
            TP += 1
        elif user[i] == 1 and scores[i] == 0:
            FN += 1
        elif user[i] == 0 and scores[i] == 0:
            TN += 1
        else:
            FP += 1
    if TP > 0:
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)
    return [precision, recall, F1]


def splitTrainTestRow(data, rowindex):
    train = []
    test = []
    for i in range(len(data)):
        data[i].insert(0, i)
        if i in rowindex:
            test.append(data[i])
        else:
            train.append(data[i])
    return train, test


def experiment(train, test, test_items, p_dict, k, threshold=0.5):
    all_prediction = []
    all_neighbors = []
    for i in range(len(test)):
        test_index = test[i][0]
        neighbors = findNeighbors(test[i], train, k, p_dict)
        all_pred = []
        for j in range(len(neighbors)):
            scores = computeRecommend(test_items[test_index], neighbors[j], test_items, threshold)
            prediction = predictCompare(test_items[test_index], scores)
            all_pred.append(prediction)
        all_prediction.append(all_pred)
        all_neighbors.append(neighbors)

    return all_prediction, all_neighbors


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
    std0 = np.std(results[0])
    std1 = np.std(results[1])
    std2 = np.std(results[2])
    pvalue01 = scipy.stats.ttest_rel(results[0], results[1])[1]
    pvalue02 = scipy.stats.ttest_rel(results[0], results[2])[1]
    pvalue12 = scipy.stats.ttest_rel(results[1], results[2])[1]

    return [avg0, avg1, avg2, std0, std1, std2, pvalue01, pvalue02, pvalue12]


if __name__ == "__main__":

    filename = "jester-data-2.csv"
    #randomly select 10000 patients for the experiment
    rowindex0 = random.sample(range(0, 23500), 1000)
    #randomly select 30 items used for testing
    columnindex0 = random.sample(range(1, 101), 30)
    data_traincol, data_test_col = readData(filename, rowindex0, columnindex0)

    #randomly select 1,000 patients for testing
    rowindex1 = random.sample(range(0, 1000), 100)

    training_row, testing_row = splitTrainTestRow(data_traincol, rowindex1)
    # training_row_df = pd.DataFrame(training_row)
    # training_row_df.describe()


    p_dict = varDistribution(training_row)
    k = 20
    all_predictions, all_neighbors = experiment(training_row, testing_row, data_test_col, p_dict, k, threshold=0.3)
    'done!'
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

