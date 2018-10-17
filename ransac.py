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
    sumdist = 0
    try:
        for i in range(0, len(X) - 1):

            # Distance from point to line
            distance = abs(k_b[0]*X[i] - y[i] + k_b[1])/math.sqrt(k_b[0]**2 + 1)
            if distance <= inlier_threshold:
                count = count + 1
                sumdist += abs(k_b[0]*X[i] - y[i] + k_b[1])/math.sqrt(k_b[0]**2 + 1)
        return count, sumdist/count
    except:
        return 0, inlier_threshold


def ransac(X, y, max_iters, inlier_threshold, min_inliers=6):
    best_model = None
    # best_model_performance = 0
    best_average_dist = inlier_threshold

    index = X.shape[0]

    for i in range(max_iters):
        sample = np.random.choice(index, size=2, replace=False)
        model_params = least_squares([X[sample[0]], y[sample[0]]], [X[sample[1]], y[sample[1]]])

        model_performance, average_distance = evaluate_model(X, y, model_params, inlier_threshold)
        if model_performance < min_inliers:
            continue

        #if model_performance > best_model_performance and avarage_distance < best_avarage_dist:
        if average_distance <= best_average_dist:
            best_model = model_params
            # best_model_performance = model_performance
            best_average_dist = average_distance
    return best_model, best_average_dist



def main():

    name_file = './Data/im_3.npy'
    data, image = HT(name_file)

    max_iterations = 500
    inlier_threshold = 0.1

    # plot points
    #plt.scatter(data.T[0], data.T[1])

    # all best lines
    lines = []

    count_lines = 0

    # average distances of all the best lines
    distance = []

    Flag = True

    # the number of iterations to search for a new best line
    count_iterations = 0
    while Flag == True:

        # RANSAC
        if count_lines == 0:
            model, dist = ransac(np.asarray(data[:, 1]), np.asarray(data[:, 0]), max_iterations, inlier_threshold)
        else:
            model, dist = ransac(np.asarray(data[:, 1]), np.asarray(data[:, 0]), max_iterations, sum(distance)/len(distance))

        # If RANSAC didn't find the best model
        if model == None:
            print('NONE')
            Flag = False
            continue

        # Coordinates of all the best lines
        all_x = []
        for z in range(0, len(data) - 1):
            # Distance from point to line
            d = abs(model[0] * data[:, 1][z] - data[:, 0][z] + model[1]) / math.sqrt(model[0] ** 2 + 1)
            if d <= inlier_threshold:
                all_x.append(data[:, 1][z])

        my_detect = 0
        if count_lines >= 1:
            # We check whether such a line already exists
            for line in lines:
                if len(set(line) & set(all_x)) >= 2:
                    my_detect += 1
            if my_detect > 0:
                count_iterations += 1
                print(count_iterations)
                if count_iterations == int(len(data)/count_lines):
                    Flag = False
                continue
            else:
                lines.append(all_x)
                distance.append(dist)
        else:
            lines.append(all_x)
            distance.append(dist)

        # plot lines
        y = []
        x = []
        for j in range(min(all_x), max(all_x)):
                y.append(j*model[0] + model[1])
                x.append(j)
        plt.plot(x, y, color=(0, 1, 0))
        count_lines += 1
        count_iterations = 0
    plt.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    main()