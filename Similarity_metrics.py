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

    def goodall3(self, x, y, p):
        if x == y:
            self.res2 = 1 - p[int(x)] ** 2
        return self.res2

    def eskin(self, x, y, n=2):
        """
        :param x:
        :param y:
        :param n: number of possible values of this attribute
        :return:
        """
        if x == y:
            self.res3 = 1
        else:
            self.res3 = float(n**2) / (n**2 + 2)
        return self.res3


def prepareAttribute(data1, data2, ps):
    all_sim1 = []
    all_sim2 = []
    all_sim3 = []
    for d in range(1, len(data1)):
        simMtric = simMetrics()
        all_sim1.append(simMtric.overlap(data1[d], data2[d]))
        all_sim2.append(simMtric.goodall3(data1[d], data2[d], ps[d]))
        all_sim3.append(simMtric.eskin(data1[d], data2[d]))
    return [all_sim1, all_sim2, all_sim3]


def computeSim(all_sims):
    sim1 = float(sum(all_sims[0])) / len(all_sims[0])
    sim2 = float(sum(all_sims[1])) / len(all_sims[1])
    sim3 = float(sum(all_sims[2])) / len(all_sims[2])
    return [sim1, sim2, sim3]

if __name__ == "__main__":

    user = [0, 1, -1, 0, 1]
    train = [0, 0, 1, -1, 1]

    simMtric = simMetrics()

    for i in range(0,3):
        simMtric = simMetrics()
        print simMtric.overlap(user[i], train[i])
        print simMtric.goodall3(user[i], train[i])
        print simMtric.eskin(user[i], train[i])

    sims = prepareAttribute(user, train)
    weighted_sims = computeSim(sims)
    print sims







