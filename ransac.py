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
    #points_in_line = []

    for i in range(0, len(X) - 1):
        distance = abs(k_b[0]*X[i] - y[i] + k_b[1])/math.sqrt(k_b[0]**2 + 1)
        if distance <= inlier_threshold:
            count = count + 1
            #points_in_line.append((y[i], X[i]))
    # return count, points_in_line
    return count


def ransac(X, y, max_iters, inlier_threshold, min_inliers):
    best_model = None
    best_model_performance = 0
    index = X.shape[0]

    # points_in_line = []

    for i in range(max_iters):
        sample = np.random.choice(index, size=2, replace=False)
        model_params = least_squares([X[sample[0]], y[sample[0]]], [X[sample[1]], y[sample[1]]])

        model_performance = evaluate_model(X, y, model_params, inlier_threshold)

        if model_performance < min_inliers:
            continue

        if model_performance > best_model_performance:
            best_model = model_params
            best_model_performance = model_performance

    # return best_model, points_in_line
    return best_model



def main():
    max_iterations = 100
    inlier_threshold = 1
    min_inliers = 8
    """
     data = [x, y]
    """

    data, image = HT()

    #data = np.load('Data/toy-data-ransac.npy')

    # plot points
    #plt.scatter(data.T[0], data.T[1])

    lines = []

    i = 0
    while i < int(math.log(len(data))):
        # our RANSAC
        model = ransac(np.asarray(data[:, 1]), np.asarray(data[:, 0]), max_iterations, inlier_threshold,
                                 min_inliers)

        xx = []
        for k in range(0, len(data) - 1):
            distance = abs(model[0] * data[:, 1][k] - data[:, 0][k] + model[1]) / math.sqrt(model[0] ** 2 + 1)
            if distance <= inlier_threshold:
                xx.append(data[:, 1][k])

        my_detect = 0
        if i >= 1:
            for line in lines:
                if len(set(line) & set(xx)) >= 2:
                    my_detect += 1
            if my_detect > 0:
                continue
            else:
                lines.append(xx)
        else:
            lines.append(xx)

        y = []
        x = []
        for j in range(min(xx), max(xx)):
                y.append(j*model[0] + model[1])
                x.append(j)
        plt.plot(x, y, color=(0, 1, 0))
        i += 1
        print(lines)

    plt.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    main()