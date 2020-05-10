import csv
import random
import math
def load_csv(filename):
    with open(filename) as csvfile:
        dataset = []
        reader = csv.reader(csvfile,delimiter=';')
        for row in reader:
            row[4] =  ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
            row[:4] = [float(row[j]) for j in range(len(row))]
            dataset.append(row)
    return dataset
def matrix_mul_bias(A, B, bias): # Fungsi perkalian matrix + bias (untuk Testing)
    C = [[0 for i in xrange(len(B[0]))] for i in xrange(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C
def sigmoid(h,derivative=False): #fungsi aktivasi sigmoid
    if derivative: #jika menggunakan backpro gunakan turunan dari sigmoid
        for i in range(len(h)):
            h[i] = h[i] * (1.0 - h[i])
    else:
        for i in range(len(h)):
            h[i] = 1 / (1 + math.exp(-h[i]))
    return h
def matrix_vector_bias(V,W,B):#fungsi perkalian vector dengan matrix dan bias
    result = [0 for i in range(len(W))]
    for i in range(len(W)):
        for j in range(len(W[0])):
            result[i]+=V[j] * W[i][j]
        result[i]+=B[i]
    return result
def mat_vec(A, B): # Fungsi perkalian matrix dengan vector (untuk backprop)
    C = [0 for i in xrange(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C

dataset=load_csv('Iris.csv')
random.shuffle(dataset)
datatrain = dataset[:int(len(dataset) * 0.8)]
datatest = dataset[int(len(dataset) * 0.8):]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]
#input -> hidden -> output
neuron =[4,4,3]
#inisialisasi bobot dan bias
weight_1 = [[0 for j in range(neuron[0])] for i in range(neuron[1])]
weight_2 = [[0 for j in range(neuron[1])] for i in range(neuron[2])]
bias_1 = [0 for i in range(neuron[1])]
bias_2 = [0 for i in range(neuron[2])]
#memberikan nilai random pada bobot weight_1 dan weight_2
for i in range(neuron[1]):
    for j in range(neuron[0]):
        weight_1[i][j] = 2 * random.random() - 1
for i in range(neuron[2]):
    for j in range(neuron[1]):
        weight_2[i][j] = 2 * random.random() - 1
iterasi = 400
alfa = 0.005
epoch = 400
for iter in range(400):
    cost_total = 0
    for index,data in enumerate(train_X):
        #feedforward
        h_1 = matrix_vector_bias(data,weight_1,bias_1) #input -> hidden
        X_1 = sigmoid(h_1)
        h_2 = matrix_vector_bias(X_1,weight_2,bias_2) #hidden -> output
        X_2 = sigmoid(h_2)
        # tandai class dengan nilai 1 [0,0,1]
        target = [0, 0, 0]
        target[int(train_y[index])] = 1

        # hitung error
        eror = 0
        for i in range(3):
            eror += 0.5 * (target[i] - X_2[i]) ** 2
        cost_total += eror
        #BELUM JADI

#contoh inisialisasi nilai
# weight_1[0][0]=0.13436424411240122
# weight_1[0][1]=0.8474337369372327
# weight_2[0][0]=0.2550690257394217
# weight_2[1][0]=0.4494910647887381
# bias_1=[0.763774618976614]
# bias_2 =[0.49543508709194095,0.651592972722763]
# x = [1,0]
# h_1 =  matrix_vector_bias(x,weight_1,bias_1)
# x_1 = sigmoid(h_1)
# h_2 =  matrix_vector_bias(x_1,weight_2,bias_2)
# x_2 = sigmoid(h_2)
# print(x_1)
# print(x_2)
