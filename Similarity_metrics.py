"""
Similarity metrics for categorical variables
"""
class simMetrics:

    def __init__(self):
        self.res1 = 0
        self.res2 = 0

    def overlap(self, x, y):
        if x == y:
            self.res1 = 1
        else:
            self.res1 = 0
        return self.res1

    def goodall1(self, x, y):
        if x == y:
            self.res2 = 1
        else:
            self.res2 = 0
        return self.res2


