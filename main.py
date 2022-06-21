from scipy import ndimage
from accessify import protected
import cv2 as cv
import numpy as np
import math
import sys

class Deconvolve(object):

    def __int__(self):
        pass

    '''Вычисление градиента регулязирующего функционала'''
    @protected
    def g(self, zk, u, A, a1, a2):

        '''Вычисление производной нормы разницы между исходным и искаженным изображением'''
        At = np.rot90(A, 2)
        temp = ndimage.convolve(input=zk, weights=A, mode='nearest')
        temp = temp - u
        temp = ndimage.convolve(input=temp, weights=At, mode='nearest')
        temp = np.multiply(temp, np.full((temp.shape[0], temp.shape[1]), 2))

        '''Вычисление субградиента функционала полной вариации'''
        directions = [(1, 0), (0, 1), (1, -1), (1, 1)]
        sum1 = np.zeros((zk.shape[0], zk.shape[1]))
        sum2 = np.zeros((zk.shape[0], zk.shape[1]))
        for direct in directions:
            row = direct[0]
            col = direct[1]

            image_temp = cv.copyMakeBorder(zk, 1, 1, 1, 1, cv.BORDER_REFLECT)
            t = image_temp[1 + row: image_temp.shape[0] - 1 + row, 1 + col:image_temp.shape[1] - 1 + col]

            t = t - zk

            t = np.where(t > 0, 1, t)
            t = np.where(t < 0, -1, t)
            t = np.where(t == 0, 0, t)

            image_temp = cv.copyMakeBorder(t, 1, 1, 1, 1, cv.BORDER_REFLECT)
            t1 = image_temp[1 - row: image_temp.shape[0] - 1 - row, 1 - col: image_temp.shape[1] - 1 - col]

            t = t1 - t
            t = (1 / (math.sqrt(row * row + col * col))) * t
            sum1 += t

            image_temp = cv.copyMakeBorder(zk, 1, 1, 1, 1, cv.BORDER_REFLECT)
            k = image_temp[1 + row: image_temp.shape[0] - 1 + row, 1 + col:image_temp.shape[1] - 1 + col]
            k1 = image_temp[1 - row: image_temp.shape[0] - 1 - row, 1 - col:image_temp.shape[1] - 1 - col]

            k = k + k1 - 2 * zk

            k = np.where(k > 0, 1, k)
            k = np.where(k < 0, -1, k)
            k = np.where(k == 0, 0, k)

            image_temp = cv.copyMakeBorder(k, 1, 1, 1, 1, cv.BORDER_REFLECT)
            k1 = image_temp[1 + row: image_temp.shape[0] - 1 + row, 1 + col:image_temp.shape[1] - 1 + col]
            k2 = image_temp[1 - row: image_temp.shape[0] - 1 - row, 1 - col:image_temp.shape[1] - 1 - col]

            k2 = k1 + k2 - 2 * k

            k2 = (1 / (math.sqrt(row * row + col * col))) * k2
            sum2 += k2

        sum = a1 * sum1 + a2 * sum2
        return temp + sum

    '''Коэффициент в методе ускоренного градиента Нестерова'''
    @protected
    def bk1(self, i, gk, zk, n):
        return 25 * math.pow(0.01, float(i) / n) * zk.shape[0] * zk.shape[1] / np.sum(np.abs(gk))

    '''Метод ускоренного градиента Нестерова'''
    def processing(self, u, m, a1, a2, n, A):
        zk = np.zeros((u.shape[0], u.shape[1]))
        vk = 0
        for i in range(1, n):
            gk1 = self.g(zk + m * vk, u, A, a1, a2)
            vk = m * vk - self.bk1(i, gk1, zk, n) * gk1
            zk = zk + vk

        zk = zk.astype(np.uint8)
        return zk

'''Разбор входной строки'''
path = sys.argv[1]
img = cv.imread(path, cv.IMREAD_GRAYSCALE).astype(np.float64)
kernel = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE).astype(np.float64)
kernel = kernel / np.sum(kernel)
sd = float(sys.argv[4])
obj = Deconvolve()
res = obj.processing(img, 0.8, math.log2(sd + 1) + 0.2, 0.001, int(3 * sd) + 40, kernel)
cv.imwrite(sys.argv[3], res)
