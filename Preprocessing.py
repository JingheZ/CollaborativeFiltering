"""
Preprocess the Jester Dataset for collaborative filtering
"""
import random
import numpy as np
import pandas as pd
import simMetrics, simInstance from Similarity_metrics


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

    return neighbors_overlap, neighbors_goodall, neighbors_eskin


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











if __name__ == "__main__":

    filename = "jester-data-2.csv"
    #randomly select 10000 patients for the experiment
    rowindex0 = random.sample(range(0, 23500), 10000)
    #randomly select 30 items used for testing
    columnindex0 = random.sample(range(1, 101), 30)
    data_traincol, data_test_col = readData(filename, rowindex0, columnindex0)

    #randomly select 1,000 patients for testing
    rowindex1 = random.sample(range(0, 10000), 1000)

