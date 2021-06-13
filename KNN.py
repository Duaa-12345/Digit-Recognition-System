
import csv
import numpy as np
import operator


def CalculateEuclideanDistance(test, train, length):     #Euclidean Distance
    distance = 0
    for i in range(length-1):
        distance +=(test[i] - train[i])**2
    Euclidean_distance = distance**(1/2)
    return Euclidean_distance



def findNeighbours(train, test, k):    #finding K neighbours
    N_distance = []
    for i in range(len(train)):
        R_Distance = CalculateEuclideanDistance(test, train[i, 1:785], len(test))

        # Storing distances w.r.t train matrix rows
        N_distance.append((train[i], R_Distance))

    N_distance.sort(key=operator.itemgetter(1))

    # Storing 1st "K" distances
    K_neighbors = []
    for i in range(k):
        K_neighbors.append(N_distance[i][0])
    return K_neighbors


def findBestNeighbour(find_neighbours):         #Choosing best neighbour
    N_count = {}
    #Finding neighbour with maximum occurance
    for i in range(len(find_neighbours)):
        occurrence =find_neighbours[i][0]
        if occurrence in N_count:
            N_count[occurrence] += 1
        else:
            N_count[occurrence] = 1
    B_Neighbour = sorted(N_count.items(), key=operator.itemgetter(1), reverse=True)
    return B_Neighbour[0][0]


def main():
    # opening csv file
    with open('train.csv', newline='') as csv_file1:
        train_data_lines = csv.reader(csv_file1)
        train_dataset = list(train_data_lines)
        train_matrix = np.array(train_dataset).astype("int")   # Converting list into matrix and changing Datatype into int

    with open('sample.csv', newline='') as csv_file2:
        test_data_lines  = csv.reader(csv_file2)
        test_dataset = list(test_data_lines)
        test_matrix = np.array(test_dataset).astype("int")


    predictions = []       # This will contain predicted values
    k = 1
    for i in range (len(test_dataset)):
        find_neighbours = findNeighbours(train_matrix, test_matrix[i], k)
        result = findBestNeighbour(find_neighbours)
        predictions.append(result)
        print('Original Number is:' + repr(test_matrix[i, 0]) + ' Prediction is:' + repr(result))

    # Finding the accuracy
    true_postives = 0
    for i in range(len(test_matrix)):
        if test_matrix[i][0] == predictions[i]:
            true_postives += 1

    accuracy = (true_postives / len(test_matrix)) * 100.0
    print('Accuracy: ' + repr(accuracy) + '%')

main()