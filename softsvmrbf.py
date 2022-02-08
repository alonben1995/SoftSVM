import itertools

import matplotlib
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt

solvers.options['show_progress'] = False


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma))

def softsvmbf(l: float, sigma: float, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m = np.shape(trainX)[0]

    #defining the graham matrix
    G = np.zeros((m, m))
    for i, j in itertools.product(range(m), range(m)):
        G[i, j] = gaussian_kernel(trainX[i],trainX[j],sigma)

    H = np.block([[2*l*G          ,np.zeros((m,m))],
                  [np.zeros((m,m)),np.zeros((m,m))]])
    eps=0.0000001
    eps_matrix=np.diag(eps*np.ones(2*m))
    sing_H=sparse(matrix(H+eps_matrix))
    u = matrix(np.hstack((np.zeros(m), (1 / m) * np.ones(m))))

    A_lt = np.zeros((m, m))
    for i in range(m):  # build A left top block
        A_lt[i] = trainy[i] * G[i]

    A = sparse(matrix(np.block([[A_lt            , np.eye(m)],
                         [np.zeros((m, m)), np.eye(m)]])))

    v = matrix(np.hstack((np.ones(m),np.zeros(m))))

    sol = solvers.qp(sing_H, u, -A, -v)
    alphas = np.array(sol['x'][:m])
    return alphas.reshape((m, 1))


def predict_softsvmbf(alphas, sigma, trainX, to_predict):
    # column j is all that's needed to calculate the prediction for example j
    G_prediction = np.array([[alphas[j].item() * gaussian_kernel(trainX[j], to_predict[i], sigma)
                             for i in range(to_predict.shape[0])]
                             for j in range(trainX.shape[0])])
    return np.sign(np.sum(G_prediction, axis=0))


def k_fold(lambdas, sigmas, k):
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    m = trainX.shape[0]
    split_trainX = np.split(trainX, k)
    split_trainy = np.split(trainy, k)

    lowest_err = (np.inf, 0)  # value, index
    params_product = list(itertools.product(lambdas, sigmas))
    for j, (l, sigma) in enumerate(params_product):
        errors = []
        for i in range(k):
            _validation_x = split_trainX[i]
            _validation_y = split_trainy[i]
            _trainX = np.concatenate([a for p, a in enumerate(split_trainX) if p != i])
            _trainy = np.concatenate([a for p, a in enumerate(split_trainy) if p != i])

            alphas = softsvmbf(l, sigma, _trainX, _trainy)
            pred = predict_softsvmbf(alphas, sigma, _trainX, _validation_x)
            new_err = np.mean(_validation_y != pred.reshape((pred.shape[0], 1)))
            errors.append(new_err)

        err=np.mean(errors)
        print(f'err avg for λ={l:3}, σ={sigma:4} is {err}')
        if err < lowest_err[0]:
            lowest_err = (err, j)

    print(f'lowest error ({lowest_err[0]}) achieved with (λ,σ)={params_product[lowest_err[1]]}')

    print(f"Testing with (λ,σ)={params_product[lowest_err[1]]}...")
    l = params_product[lowest_err[1]][0]
    sigma = params_product[lowest_err[1]][1]
    alphas = softsvmbf(l, sigma, trainX, trainy)
    pred = predict_softsvmbf(alphas, sigma, trainX, testX)
    err = np.mean(testy != pred.reshape((pred.shape[0], 1)))
    print(f"Test error with (λ,σ)={params_product[lowest_err[1]]}: {err}")


def paint():
    """
    paints the train data points by labels
    """
    num_labels = 2
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain'].flatten()
    colorsxy = [([], []) for _ in range(num_labels)]
    for i in range(trainX.shape[0]):
        colorsxy[max(trainy[i], 0)][0].append(trainX[i, 0])
        colorsxy[max(trainy[i], 0)][1].append(trainX[i, 1])
    for color in colorsxy:
        plt.scatter(color[0], color[1], s=1)
    plt.show()


def paint_classifier(l, sigmas):
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    x_min, x_max = np.min(trainX[:, 0]), np.max(trainX[:, 0])
    y_min, y_max = np.min(trainX[:, 1]), np.max(trainX[:, 1])
    x_space = np.linspace(x_min, x_max, 100)
    y_space = np.linspace(y_min, y_max, 100)
    prod = np.array(list(itertools.product(x_space, y_space)))

    for sigma in sigmas:
        alphas = softsvmbf(l, sigma, trainX, trainy)
        pred = predict_softsvmbf(alphas, sigma, trainX, prod)

        colorsxy = ([], [])
        for i in range(len(pred)):
            colorsxy[max(int(pred[int(i)]), 0)].append(prod[i])
        for color in colorsxy:
            if color:
                color = np.row_stack(color)
                plt.scatter(color[:, 0], color[:, 1], s=10)
        plt.title(fr"$\lambda={l}, \sigma={sigma}$")
        plt.show()

def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvmbf(10, 0.1, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"



if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    paint()
    # k_fold([1, 10, 100], [0.01, 0.5, 1], 5)
    # k_fold([1], [0.5], 5)

    paint_classifier(100, [0.01, 0.5, 1])
