import random
import numpy as np
from harris_toy import main as HT
from matplotlib import pyplot as plt
from sklearn import linear_model

def least_squares(X, y):
    """
    y = mx + c
    """
    # b = np.ones((X.shape[0], 1))
    # A = np.hstack((X, b))
    # theta = np.linalg.lstsq(A, y)[0]

    A = np.vstack([X, np.ones(len(X))]).T
    #print(A)
    m_c = np.linalg.lstsq(A, y, rcond=None)[0]
    #print(m_c)
    return m_c


def evaluate_model(X, y, m_c, inlier_threshold):
    # #b = np.ones((X.shape[0], 1))
    # b = np.ones(len(X))
    # #y = y.reshape((y.shape[0], 1))
    # y = y.T
    # A = np.hstack((y, X, b)).T
    # theta = np.insert(m_c, 0, -1.)
    # print(theta)
    #
    # distances = np.abs(np.sum(A * theta, axis=1)) / np.sqrt(np.sum(np.power(theta[:-1], 2)))
    # inliers = distances <= inlier_threshold
    # num_inliers = np.count_nonzero(inliers == True)
    # print(num_inliers)
    # return num_inliers


    count = 0
    for i in range(0, len(X)):
        if (y[i] - m_c[0]*X[i] + m_c[1]) <= inlier_threshold:
            count = count + 1
    #print(count)
    return count


def ransac(X, y, max_iters=10000, samples_to_fit=30, inlier_threshold=0.01, min_inliers=30):
    best_model = None
    best_model_performance = 0

    num_samples = X.shape[0]

    for i in range(max_iters):
        sample = np.random.choice(num_samples, size=samples_to_fit, replace=False)
        model_params = least_squares(X[sample], y[sample])
        model_performance = evaluate_model(X, y, model_params, inlier_threshold)

        if model_performance < min_inliers:
            continue

        if model_performance > best_model_performance:
            best_model = model_params
            best_model_performance = model_performance

    return best_model


def func(Data):
    X = Data[:, 1]
    y = Data[:, 0]
    X = X.reshape(-1, 1)
    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(X, y)

    # Predict data of estimated models
    line_X = np.arange(0, 10)
    line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])
    plt.plot(line_X, line_y_ransac, '-b')


#xys, image = HT()

# test data
n = 100
xys = np.random.random((n, 2)) * 10
xys[:50, 1:] = xys[:50, :1]

# точки на графік
plt.scatter(xys.T[0], xys.T[1])


# наша ф-я
model = ransac(np.asarray(xys[:, 1]), np.asarray(xys[:, 0]))

# встроєна
q = func(xys)


y = []
x = []
for i in range(0, 10):
    #if (i*model[0] + model[1])> 0 and (i*model[0] + model[1])< 200:
    y.append(i*model[0] + model[1])
    x.append(i)
plt.plot(x, y, color=(0, 1, 0))

#plt.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
plt.show()