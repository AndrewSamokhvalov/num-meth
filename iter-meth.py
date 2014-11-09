__author__ = 'andrey'

import numpy as np
import time

from math import *
from matplotlib import pyplot as plt

file_A = 'A.txt'
file_b = 'b.txt'
delimiter = ','



class Info():

    iteration = 0
    delay = 0
    xk = 0
    eps = 0

    def description(self):
        print("iter count: %s" % self.iteration)
        print("xk: %s" % self.xk)
        print("delay: %s" % self.delay)

class SolvingMethod():
    def isQuad(self, matr):
        return matr.shape[1] == matr.shape[0]

    def isSymetric(self, matr):
        if not self.isQuad(matr):
            return False

        n = matr.shape[1]
        transpose = matr.transpose()
        answer = transpose - matr

        for i in range(1, n):
            for j in range(1, n):
                if answer[i][j] != 0:
                    return False
        return True

    def isconvergence(self, mB):
        return max(sum(abs(mB))) < 1


class QR(SolvingMethod):
    def get_beta(self, a, b):
        return np.dot(a, b) / np.dot(a, a)

    def solve(self, mA, b):

        # bi = mA(,)
        pass


class IIM(SolvingMethod):
    def get_r(self, xk_1, matr, b):
        return np.dot(matr, xk_1) - b

    def get_t(self, b, matr):
            a = np.dot(matr, b)

            up = np.dot(a.transpose(), b)
            down = np.dot(a.transpose(), a)
            return up / down

    def solve(self, mA, b, eps):
        info = Info()
        info.eps = eps

        start = time.time()
        if not self.isSymetric(mA):
            b = np.dot(mA.transpose(), b)
            mA = np.dot(mA.transpose(), mA)

        xk = b.copy()
        answer = np.dot(np.linalg.inv(mA), b)
        dif = [eps + 1]
        while max(dif) > eps:
            rk = self.get_r(xk, mA, b)
            tk_1 = self.get_t(rk, mA)

            xk = xk - tk_1 * rk
            dif = answer - xk
            info.iteration += 1


        info.xk = xk
        info.delay = time.time() - start

        return info


iim = IIM()
qr = QR()
eps = 1.0 / pow(10.0, 0)


linfo = []
for n in range(3, 30):

    mA = np.random.rand(n, n)
    b = np.random.rand(n)
    linfo.append(iim.solve(mA, b, eps))
    print("%s : %s" % (n, np.linalg.cond(mA)))


plt.plot(list(map(lambda info: info.delay, linfo)))
plt.show()





