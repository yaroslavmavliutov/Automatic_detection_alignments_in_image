import numpy as np
from matplotlib import pyplot as plt
from harris_toy import harris_toy as HT
from sklearn import linear_model, datasets
from skimage.measure import ransac
from skimage.feature import plot_matches
import random

#1, 0
coord, image = HT()
N = 5
count = int(coord.shape[0]/N)
xx = np.empty(count)
yy = np.empty(count)
for i in range(N):
    for i in range(0, count):
        index = random.randint(0, coord.shape[0] - 1)
        xx = np.insert(xx, i, coord[index][1])
        yy = np.insert(yy, i, coord[index][0])
    # print(test)
    # xx = test[:, 1]
    # yy = test[:, 0]

    xx = xx[~np.isnan(xx)]
    yy = yy[~np.isnan(yy)]

    y = yy.tolist()
    y = np.asarray(y)
    X = xx.reshape(-1, 1)

    plt.plot(X, y, '.')
    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(X, y)
    inlier_mask = model_ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(0, 300)
    line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])


    plt.plot(X[inlier_mask], y[inlier_mask], '.g')
    plt.plot(X[outlier_mask], y[outlier_mask], '.r')
    plt.plot(line_X, line_y_ransac, '-b')
    plt.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    plt.legend(loc='lower right')
plt.show()


def ransac(X, y, fit_fn, evaluate_fn, max_iters=100, samples_to_fit=2, inlier_threshold=0.1, min_inliers=10):
    best_model = None
    best_model_performance = 0

    num_samples = X.shape[0]

    for i in range(max_iters):
        sample = np.random.choice(num_samples, size=samples_to_fit, replace=False)
        model_params = fit_fn(X[sample], y[sample])
        model_performance = evaluate_fn(X, y, model_params, inlier_threshold)

        if model_performance < min_inliers:
            continue

        if model_performance > best_model_performance:
            best_model = model_params
            best_model_performance = model_performance

    return best_model