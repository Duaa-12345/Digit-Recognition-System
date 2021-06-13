from _ast import In

import inline as inline
import matplotlib
import pandas as pd
import numpy as np  # numerical computing
from scipy.optimize import minimize  # optimization code
import matplotlib.pyplot as plt  # plotting
import seaborn as sns

sns.set()
import itertools

"reading CSV files"
data_full = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

"returns top 10 rows of data ( i.e. only for testing)"
data_full.head(10)

# print(data_full)
# print(data_test)

"putting all the entries to matrix"
data_full_matrix = data_full.to_numpy()

"There is column in Y for each of the digits 0-9, 42000 rows in Y are present"
Y = np.zeros((data_full_matrix.shape[0], 10))
for i in range(10):
    Y[:, i] = np.where(data_full_matrix[:, 0] == i, 1, 0)

"separating label column"
y_labels, data_09 = data_full_matrix[:, 0], data_full_matrix[:, 1:]

"separating columns that contain all 0 values"
data_09 = data_09[:, data_09.sum(axis=0) != 0]

"training , testing and validating data sets"
data_train_09, Y_train_09 = data_09[0:29400, :], Y[0:29400, :]
data_val_09, Y_val_09 = data_09[29400:, :], Y[29400:, :]
data_test_09, Y_test_09 = data_09[35700:, :], Y[35700:, :]

y_labels_train = y_labels[0:29400]
y_labels_val = y_labels[29400:35700]
y_labels_test = y_labels[35700:]


# print(data_train_09.shape, Y_train_09.shape)
# print(data_val_09.shape, Y_val_09.shape)
# print(data_test_09.shape, Y_test_09.shape)

def multinomial_partitions(n, k):
    """returns an array of length k sequences of integer partitions of n"""
    nparts = itertools.combinations(range(1, n + k), k - 1)
    tmp = [(0,) + p + (n + k,) for p in nparts]
    sequences = np.diff(tmp) - 1
    return sequences[::-1]  # reverse the order


"adding bias (columns of 1) to the data"


def make_multinomial_features(feature_vectors, order=[1, 2]):
    X_temporary = np.ones_like(feature_vectors[:, 0])
    for ORDER in order:
        if ORDER == 1:
            f = feature_vectors
        else:
            p = multinomial_partitions(ORDER, feature_vectors.shape[1])
            f = np.column_stack((np.prod(feature_vectors ** p[i, :], axis=1) for i in range(p.shape[0])))

        X_temporary = np.column_stack((X_temporary, f))
    return X_temporary


def mean_normalize(X):
    '''apply mean normalization to each column of the matrix X'''
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    return (X - X_mean) / X_std


def apply_normalizer(X, X_mean, X_std):
    return (X - X_mean) / X_std


"data sets are normalized using the mean and standard deviation from the data set containing 42000 entries"
X_mean = data_09.mean(axis=0)
X_std = data_09.std(axis=0)
X_std[X_std == 0] = 1.0  # if there are any 0 values in X_std set them to 1

order = [1]

X_train = make_multinomial_features(data_train_09, order=order)
X_train[:, 1:] = apply_normalizer(X_train[:, 1:], X_mean, X_std)
Y_train = Y_train_09

X_val = make_multinomial_features(data_val_09, order=order)
X_val[:, 1:] = apply_normalizer(X_val[:, 1:], X_mean, X_std)
Y_val = Y_val_09

X_test = make_multinomial_features(data_test_09, order=order)
X_test[:, 1:] = apply_normalizer(X_test[:, 1:], X_mean, X_std)
Y_test = Y_test_09

" training loop implementation"

"regularization term"
reg = 300.0
np.random.seed(42)
" random guess for parameters"
guess = np.random.randn(X_train.shape[1])
" optimized parameter's matrix"
A_opt = np.zeros((X_train.shape[1], 10))
" resultant matrix to hold optimization output for each of the model"
Result = []

"Sigmoid function"


def g(z):
    return 1.0 / (1.0 + np.exp(-z))


"Model function"


def h_logistic(X, a):
    return g(np.dot(X, a))


"Function for Cost calculation"


def J(X, a, y):
    m = y.size
    return -(np.sum(np.log(h_logistic(X, a))) + np.dot((y - 1).T, (np.dot(X, a)))) / m


"Cost Function with Regularization"


def J_reg(X, a, y, reg_lambda):
    m = y.size
    return J(X, a, y) + reg_lambda / (2.0 * m) * np.dot(a[1:], a[1:])


"Gradient Cost Function"


def gradJ(X, a, y):
    m = y.size
    return (np.dot(X.T, (h_logistic(X, a) - y))) / m


"Gradient of Cost Function with Regularization"


def gradJ_reg(X, a, y, reg_lambda):
    m = y.size
    return gradJ(X, a, y) + reg_lambda / (2.0 * m) * np.concatenate(([0], a[1:])).T


"Each model is fit to it's number (0-9) by evaluation it's cost function against all of the other numbers "

for i in range(10):
    print('\nFitting {} against the rest\n'.format(i))


    def opt_J_reg(a):
        return J(X_train, a, Y_train[:, i])


    def opt_gradJ_reg(a):
        return gradJ_reg(X_train, a, Y_train[:, i], reg)


    res = minimize(opt_J_reg, guess, method='CG', jac=opt_gradJ_reg, tol=1e-6, options={'disp': True})
    Result.append(res)
    A_opt[:, i] = res.x

"Function will return the probabilities predicted by each of the models for some given input image"


def predict(sample, sample_label):
    print('\nTest sample is : {}\n'.format(sample_label))
    prob = np.zeros((10, 2))
    for num in range(10):
        a_opt = A_opt[:, num]
        prob[num, 0] = num
        prob[num, 1] = h_logistic(sample, a_opt)

    " put the best guess at the top"
    prob = prob[prob[:, 1].argsort()[::-1]]
    print('Predicted probabilities of model:\n')
    for i in range(10):
        print("{} with probability = {:.3f}".format(int(prob[i, 0]), prob[i, 1]))


" testing some random samples"

"testing 8"
sample1 = 23
sample1_label = y_labels_test[sample1]
sample = X_test[sample1, :]
predict(sample, sample1_label)

"testing 2"
sample2 = 147
# sample2_label = y_labels_test[sample2]
# sample = X_test[sample2, :]
# predict(sample, sample2_label)

"Function will give a printout of the percentage of correct prediction in the data-set."


def print_num_correct(datasets):
    for dataset in datasets:
        set_name, yl, Xd = dataset
        yls = yl.size
        prob = np.zeros((10, 2))
        count = 0
        for sample in range(yls):
            for num in range(10):
                a_opt = A_opt[:, num]
                prob[num, 0] = num
                prob[num, 1] = h_logistic(Xd[sample, :], a_opt)

            prob = prob[prob[:, 1].argsort()[::-1]]
            if prob[0, 0] == yl[sample]:
                count += 1
        print('\n{}'.format(set_name))
        print("{} correct out of {} : {}% correct".format(count, yls, count / yls * 100))


datasets = [('Training Set:', y_labels_train, X_train), ('Validation Set:', y_labels_val, X_val)]
print_num_correct(datasets)

print_num_correct([('Test Set:', y_labels_test, X_test)])

"----------------------------------------------------------------------------------------------------------------------"
