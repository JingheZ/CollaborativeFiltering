"""
Similarity metrics for categorical variables
author: JingheZ
"""
class simMetrics:
    """
    Compute the per-attribute similarity
    """

    def __init__(self):
        """
        :param d: number of attributes
        :return:
        """
        self.res1 = 0
        self.res2 = 0
        self.res3 = 0

    def overlap(self, x, y):
        if x == y:
            self.res1 = 1
        return self.res1

    def goodall1(self, x, y):
        if x == y:
            self.res2 = 1
        return self.res2

    def eskin(self, x, y, n):
        """
        :param x:
        :param y:
        :param n: number of possible values of this attribute
        :return:
        """
        if x == y:
            self.res = 1
        else:
            self.res3 = float(n**2) / (n**2 + 2)
        return self.res3


class SimInstance(object):
    """
    Compute the aggregated similarity between instances over all attributes
    """
    def __init__(self):
        self.sim1 = 0
        self.all_sim1 = []
        self.sim2 = 0
        self.all_sim2 = []
        self.sim3 = 0
        self.all_sim3 = []

    def prepareAttribute(self, data1, data2):
        for d in range(len(data1)):
            self.all_sim1.append(self.overlap(data1[d], data2[d]))
            self.all_sim2.append(self.goodall1(data1[d], data2[d]))
            self.all_sim3.append(self.eskin(data1[d], data2[d]))
        return self

    def computeSim(self):
        self.sim1 = float(sum(self.all_sim1)) / len(self.all_sim1)
        self.sim2 = float(sum(self.all_sim2)) / len(self.all_sim2)
        self.sim3 = float(sum(self.all_sim3)) / len(self.all_sim3)
        return self







