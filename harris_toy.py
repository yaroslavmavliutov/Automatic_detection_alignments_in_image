from matplotlib import pyplot as plt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from PIL import Image


def del_nan(x):
    x_new = []
    for i in range(len(x)):
        if not np.isnan(x[i][0]):
            x_new.append(x[i])

    return x_new



def my_round(x):
    x = del_nan(x)
    round_x = []

    for i in x:
        l = int(Decimal(str(i[0])).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        r = int(Decimal(str(i[1])).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        round_x.append((l, r))
    return round_x


def del_dubl(x):
    tmp = set(x)
    return np.array(list(tmp))



def point_intere(image_name):

    # image = np.load(image_name)
    image = image_name
    c_h = corner_harris(image, k = 0.06)

    coords = corner_peaks(c_h, min_distance=2)
    # coords_subpix = corner_subpix(image, coords, window_size=11)

    return del_dubl(my_round(coords))


def main():
    # image=np.load('./Data/toy-data-ransac.npy')
    image = np.load('./Data/im_2.npy')
    # c_h = corner_harris(image)

    cc = point_intere('./Data/im_2.npy')
    fig, ax = plt.subplots()
    # ax.imshow(c_h, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(cc[:, 1], cc[:, 0], '.b', markersize=8)
    return cc, image
