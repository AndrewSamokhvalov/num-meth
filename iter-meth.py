__author__ = 'andrey'

import numpy as np
import time

from math import *
from matplotlib import pyplot as plt

class Info():
    iteration = 0
    delay = 0
    eps = 0
    answer = 0
    difference = 0

    def __add__(self, other):
        info = Info()
        info.iteration = (self.iteration + other.iteration) / 2
        info.delay = (self.delay, other.delay) / 2
        info.iteration = (self.iteration + other.iteration) / 2
        info.eps = (self.eps + other.eps) / 2
        info.difference = (self.difference + other.difference) / 2
        return info



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
    def _solve(self, A, Q, R, i):
        vai = A[:, i]
        vui = vai

        for j in range(0, i):
            vej = Q[:, j]

            c = np.dot(vej, vai)
            vui = vui - c * vej
            R[j, i] = c

        nvui = np.linalg.norm(vui)

        vei = vui / nvui
        Q[:, i] = vei
        R[i, i] = np.dot(vei, vai)

    def solve(self, A, b):
        info = Info()
        start = time.time()

        [n, n] = A.shape

        Q = np.zeros((n, n))
        R = np.zeros((n, n))

        for i in range(0, n):
            self._solve(A, Q, R, i)

        Q = - Q
        R = - R

        info.answer = np.dot(np.dot(np.linalg.inv(R), Q,), b)
        info.delay = time.time() - start
        return info


class IIM(SolvingMethod):
    def get_r(self, xk_1, matr, b):
        return np.dot(matr, xk_1) - b

    def get_t(self, b, matr):
        a = np.dot(matr, b)

        return np.dot(a.transpose(), b) / np.dot(a.transpose(), a)

    def solve(self, mA, b):
        info = Info()
        info.eps = pow(10, -3)

        start = time.time()
        if not self.isSymetric(mA):
            b = np.dot(mA.transpose(), b)
            mA = np.dot(mA.transpose(), mA)

        xk = b.copy()

        isContinue = True
        while isContinue:
            rk = self.get_r(xk, mA, b)
            tk_1 = self.get_t(rk, mA)

            prev_xk = xk.copy()
            xk = xk - tk_1 * rk
            info.iteration += 1

            isContinue = info.eps < np.linalg.norm(prev_xk - xk)

        info.answer = xk
        info.delay = time.time() - start

        return info


N = 500
c = 3

iim = IIM()
qr = QR()

l_info_qr = []
l_info_iim = []

def get_info(f, infos):
    return list(map(f, infos))

for n in range(3, N):
    mA = np.random.rand(n, n)
    b = np.random.rand(n)

    real_answer = np.dot(np.linalg.inv(mA), b)

    info_qr = qr.solve(mA, b)
    info_iim = iim.solve(mA, b)

    info_qr.difference = np.linalg.norm(info_qr.answer - real_answer)
    info_iim.difference = np.linalg.norm(info_iim.answer - real_answer)

    l_info_qr.append(info_qr)
    l_info_iim.append(info_iim)

    print("Progress: %.1f" % (n / N * 100), end="\r")


l_qr_delays = [info.delay for info in l_info_qr]
l_iim_delays = [info.delay for info in l_info_iim]

l_qr_err = [info.difference for info in l_info_qr]
l_iim_err = [info.difference for info in l_info_iim]

plt.title("QR IIM comparison")
plt.xlabel("N")
plt.ylabel("T(n)")

print("QR average error: %f" % (sum(l_qr_err) / len(l_qr_err)))
print("IIM average error: %f " % (sum(l_iim_err) / len(l_iim_err)))


plt.plot(l_qr_delays, 'o', label='qr delay')
plt.plot(l_iim_delays, 's', label='iim delay')

plt.show()







