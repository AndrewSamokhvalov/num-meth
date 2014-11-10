__author__ = 'andrey'

import numpy as np
import time

# from math import *
# from matplotlib import pyplot as plt

file_A = 'A.txt'
file_b = 'b.txt'
delimiter = ','
mA = np.loadtxt(file_A, delimiter=delimiter)
b = np.loadtxt(file_b, delimiter=delimiter)

class Info():
    iteration = 0
    delay = 0
    eps = 0
    answer = 0

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

        [n, n] = A.shape

        Q = np.zeros((n, n))
        R = np.zeros((n, n))

        for i in range(0, n):
            self._solve(A, Q, R, i)

        Q = - Q
        R = - R

        info.answer = np.dot(np.dot(np.linalg.inv(R), Q,), b)
        return info





class IIM(SolvingMethod):
    def get_r(self, xk_1, matr, b):
        return np.dot(matr, xk_1) - b

    def get_t(self, b, matr):
        a = np.dot(matr, b)

        return np.dot(a.transpose(), b) / np.dot(a.transpose(), a)

    def solve(self, mA, b):
        info = Info()
        info.eps = 0.1

        start = time.time()
        if not self.isSymetric(mA):
            b = np.dot(mA.transpose(), b)
            mA = np.dot(mA.transpose(), mA)

        xk = b.copy()


        prev_xk = xk + 10 * info.eps

        while info.eps < np.linalg.norm(prev_xk - xk):
            rk = self.get_r(xk, mA, b)
            tk_1 = self.get_t(rk, mA)

            prev_xk = xk.copy()
            xk = xk - tk_1 * rk
            info.iteration += 1

        info.answer = xk
        info.delay = time.time() - start

        return info


iim = IIM()
qr = QR()

for n in range(3, 20):

    mA = np.random.rand(n, n)
    b = np.random.rand(n)

    answer = np.dot(np.linalg.inv(mA), b)

    info_qr = qr.solve(mA, b)
    info_iim = iim.solve(mA, b)
    
    print("Answer")
    print(answer)

    print("QR:")
    print(info_qr.answer)

    print("IIM:")
    print(info_iim.answer)






