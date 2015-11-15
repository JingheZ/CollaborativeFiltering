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

    
class SimInstance:
    """
    Compute the aggregated similarity between instances over all attributes
    """
    def __init__(self):
        self.sim = 0

    def computeSim(self, all_sim):
        self.sim = float(sum(all_sim)) / len(all_sim)
        return self.sim







