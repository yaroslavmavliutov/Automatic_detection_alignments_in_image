import numpy as np
from harris_toy import main as HT
from matplotlib import pyplot as plt
import math

def least_squares(xy1, xy2):
    """
    xy = [x, y]
    y = mx + c
    """
    try:
        m = (xy2[1] - xy1[1])/(xy2[0] - xy1[0])
        c = (xy1[1]*xy1[0] - xy1[0]*xy2[1])/(xy2[0] - xy1[0]) + xy1[1]
        if m == float('nan') or c == float('nan'): return [0, 0]
        return [m, c]

    except: return [0, 0]

def evaluate_model(X, y, m_c, inlier_threshold):
    count = 0
    for i in range(0, len(X) - 1):
        distance = abs(m_c[0]*X[i] - y[i] + m_c[1])/math.sqrt(m_c[0]**2 + 1)
        # print('x ', X[i])
        # print('y ', y[i])
        # print('m_c ', m_c)
        # print('d ', distance)
        if distance <= inlier_threshold:
            count = count + 1
        print(count)
        print('--------------')
    return count


def ransac(X, y, max_iters=1000, inlier_threshold=20, min_inliers=16):
    best_model = None
    best_model_performance = 0
    #min_inliers = X.shape[0] - 1
    index = X.shape[0]

    for i in range(max_iters):
        sample = np.random.choice(index, size=2, replace=False)
        model_params = least_squares([X[sample[0]], y[sample[0]]], [X[sample[1]], y[sample[1]]])

        model_performance = evaluate_model(X, y, model_params, inlier_threshold)

        if model_performance < min_inliers:
            continue

        if model_performance > best_model_performance:
            best_model = model_params
            best_model_performance = model_performance

    return best_model

xys, image = HT()


# точки на графік
#plt.scatter(xys.T[0], xys.T[1])


# наша ф-я
model = ransac(np.asarray(xys[:, 1]), np.asarray(xys[:, 0]))



y = []
x = []
print(model)
for i in range(0, 300):
    if abs(i*model[0] + model[1])<= np.amax(np.asarray(xys[:, 0])):
        y.append(i*model[0] + model[1])
        x.append(i)
plt.plot(x, y, color=(0, 1, 0))

plt.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
plt.show()