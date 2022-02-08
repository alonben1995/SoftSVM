import itertools

import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
solvers.options['show_progress'] = False

# todo: complete the following functions, you may add auxiliary functions or define class to help you
def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]

def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m=np.shape(trainX)[0]
    d=np.shape(trainX)[1]
    H=sparse(matrix(np.block([[2*l*np.eye(d),np.zeros((d,m))],
                       [np.zeros((m,d)),np.zeros((m,m))]])))
    u=matrix(np.hstack((np.zeros(d),(1/m)*np.ones(m))))


    A_lt = np.zeros((m, d))
    for i in range(m):  # build A left top block
        A_lt[i] = trainy[i] * trainX[i]
    A=sparse(matrix(np.block([[A_lt,            np.eye(m)],
                       [np.zeros((m,d)), np.eye(m)]])))

    v=matrix(np.hstack((np.ones(m),np.zeros(m))))

    sol= solvers.qp(H, u, -A, -v)
    w=np.array(sol['x'][:d])
    return w

def predict_svm(w,x_test):
    ar=np.asarray([np.sign(sample@w) for sample in x_test])
    predict = np.reshape(ar, (ar.shape[0], 1))
    return predict


def k_fold(lambdas,k):
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    m = trainX.shape[0]
    split_trainX = np.split(trainX, k)
    split_trainy = np.split(trainy, k)

    validation_params = {
        'lambdas': lambdas
    }

    lowest_err = (np.inf, 0)  # value, index
    for j, l in enumerate(lambdas, 1):
        errors = []
        for i in range(k):
            _validation_x = split_trainX[i]
            _validation_y = split_trainy[i]
            _trainX = np.concatenate([a for p, a in enumerate(split_trainX) if p != i])
            _trainy = np.concatenate([a for p, a in enumerate(split_trainy) if p != i])

            w = softsvm(l,_trainX, _trainy)
            pred = predict_svm(w,_validation_x)
            new_err = np.mean(_validation_y != pred)
            errors.append(new_err)



        err = np.mean(errors)
        if err < lowest_err[0]:
            lowest_err = (err, j)
        print(f'err for lambda={l} is {err}')
    print((f'lowest error achieved with lambda={lambdas[lowest_err[1]]}'))
    return lambdas[lowest_err[1]]

def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']


    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


def main_test(powers, sample_size, repeat, title,scatter=False):


    data = np.load('EX2q2_mnist.npz')

    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    lambdas = [10 ** n for n in powers]


    avg_train_err,avg_test_err, min_test_err, max_test_err, min_train_err, max_train_err = [], [], [],[],[],[]

    for l in lambdas:
        test_means = []
        train_means=[]
        classifiers = []

        for i in range(repeat):
            print(f"lambda: {l}, repeat: {i}")
            x_train, y_train = gensmallm([trainX], [trainy], sample_size)

            classifiers.append(softsvm(l, x_train, y_train))

            print("testing...")
            for classifier in classifiers:
                pred = predict_svm(classifier, testX)
                pred = np.reshape(pred, (pred.shape[0],))
                train_pred=predict_svm(classifier, trainX)
                train_pred = np.reshape(train_pred, (train_pred.shape[0],))
                test_means.append(np.mean(testy != pred))
                train_means.append(np.mean(trainy != train_pred))

        avg_test_err.append(np.mean(test_means))
        avg_train_err.append(np.mean(train_means))
        min_test_err.append(min(test_means))
        max_test_err.append(max(test_means))
        min_train_err.append(min(train_means))
        max_train_err.append(max(train_means))
    if scatter:
        # plt.scatter(lambdas,avg_test_err, yerr=[min_test_err,max_test_err], label="avg test error",color="red")
        # plt.scatter(lambdas, min_test_err, yerr=[min_test_err, max_test_err], label="min test error", color="green")
        # plt.scatter(lambdas, max_test_err, yerr=[min_test_err, max_test_err], label="max test error", color="blue")
        #
        # plt.scatter([i+0.1*i for i in lambdas], avg_train_err, yerr=[min_train_err, max_train_err], label="train error", color="green")
        # plt.scatter([i + 0.1 * i for i in lambdas], avg_train_err, yerr=[min_train_err, max_train_err],label="train error", color="green")
        plt.scatter(lambdas,avg_test_err,color="red")
        plt.errorbar(lambdas,avg_test_err, yerr=[min_test_err,max_test_err], label="test error",color="red",fmt='o')
        plt.scatter([i+0.1*i for i in lambdas], avg_train_err,color="green")
        plt.errorbar([i + 0.1 * i for i in lambdas], avg_train_err, yerr=[min_train_err, max_train_err],label="train error", color="green",fmt='o')
    else:
        plt.errorbar(lambdas,avg_test_err, yerr=[min_test_err,max_test_err], label="test error",color="red")
        plt.errorbar([i+0.1*i for i in lambdas], avg_train_err, yerr=[min_train_err, max_train_err], label="train error", color="green")
    plt.title(title)
    plt.xlabel("lambda")
    plt.ylabel("Error")
    plt.semilogx()
    plt.xticks(lambdas)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #     simple_test()

    # here you may add any code that uses the above functions to solve question 2
    # section (a)
    #     k_fold([1,10,100],5)
        repeat = 10
        powers = [i for i in range(1,11)]
        print(powers)
        sample_size=100
        title = "soft-svm algorithm error as a function of lambda"
        xlabel = "Training sample size"
        main_test(powers,sample_size,repeat,title)

    #sample size 1000
        repeat=1
        powers=[1,3,5,8]
        sample_size=1000
        title = "soft-svm algorithm error as a function of lambda"
        main_test(powers,sample_size,repeat,title,scatter= True)