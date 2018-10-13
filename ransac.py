import numpy as np
from harris_toy import main as HT
from matplotlib import pyplot as plt
import math

def least_squares(xy1, xy2):
    """
    xy = [x, y]
    y = kx + b
    """
    try:
        k = (xy2[1] - xy1[1])/(xy2[0] - xy1[0])
        b = (xy1[1]*xy1[0] - xy1[0]*xy2[1])/(xy2[0] - xy1[0]) + xy1[1]
        if k == float('nan') or b == float('nan'): return [0, 0]
        return [k, b]

    except: return [0, 0]

def evaluate_model(X, y, k_b, inlier_threshold):
    count = 0
    for i in range(0, len(X) - 1):
        distance = abs(k_b[0]*X[i] - y[i] + k_b[1])/math.sqrt(k_b[0]**2 + 1)
        # print('x ', X[i])
        # print('y ', y[i])
        # print('m_c ', m_c)
        # print('d ', distance)
        if distance <= inlier_threshold:
            count = count + 1
    return count


def ransac(X, y, max_iters, inlier_threshold, min_inliers):
    best_model = None
    best_model_performance = 0
    index = X.shape[0]

    for i in range(max_iters):
        sample = np.random.choice(index, size=2, replace=False)
        model_params = least_squares([X[sample[0]], y[sample[0]]], [X[sample[1]], y[sample[1]]])
        #print('iteration: ', i)
        #print('koeff: ', model_params)
        model_performance = evaluate_model(X, y, model_params, inlier_threshold)
        #print('count in_point: ', model_performance)
        if model_performance < min_inliers:
            continue

        if model_performance > best_model_performance:
            best_model = model_params
            best_model_performance = model_performance

        #print('BEST: ', best_model)
        #print('-------')
    return best_model

def main():
    max_iterations = 1000
    inlier_threshold = 15
    min_inliers = 30
    """
     data = [x, y]
    """

    # data, image = HT()
    data = np.load('Data/toy-data-ransac.npy')

    # plot points
    plt.scatter(data.T[0], data.T[1])

    # our RANSAC
    model = ransac(np.asarray(data[:, 1]), np.asarray(data[:, 0]), max_iterations, inlier_threshold, min_inliers)

    # min X and max X for curve
    xx = []
    for i in range(0, len(data) - 1):
        distance = abs(model[0] * data[:, 1][i] - data[:, 0][i] + model[1]) / math.sqrt(model[0] ** 2 + 1)
        # print('x ', X[i])
        # print('y ', y[i])
        # print('m_c ', m_c)
        # print('d ', distance)
        if distance <= 15:
            xx.append(data[:, 1][i])


    y = []
    x = []
    # for i in range(np.amin(np.asarray(xys[:, 1])), np.amax(np.asarray(xys[:, 1]))):
    for i in range(min(xx), max(xx)):
        if abs(i*model[0] + model[1])<= np.amax(np.asarray(xys[:, 0])):
            y.append(i*model[0] + model[1])
            x.append(i)
    plt.plot(x, y, color=(0, 1, 0))

    #plt.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    main()