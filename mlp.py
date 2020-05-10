import csv
import random
import math
with open('Iris.csv') as csvfile:
    dataset = []
    reader = csv.reader(csvfile,delimiter=';')
    for row in reader:
        row[4] =  ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
        row[:4] = [float(row[j]) for j in range(len(row))]
        dataset.append(row)
# Split x and y (feature and target)
random.shuffle(dataset)
datatrain = dataset[:int(len(dataset) * 0.8)]
datatest = dataset[int(len(dataset) * 0.8):]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]
def matrix_mul_bias(A, B, bias): # Fungsi perkalian matrix + bias (untuk Testing)
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C
def vec_mat_bias(A, B, bias): # Fungsi perkalian vector dengan matrix + bias
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C
def mat_vec(A, B): # Fungsi perkalian matrix dengan vector (untuk backprop)
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C
def sigmoid(A, deriv=False): # Fungsi aktivasi sigmoid
    if deriv: # kalau sedang backprop pakai turunan sigmoid
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A
# Define parameter
alfa = 0.005
epoch = 400
neuron = [4, 4, 3] # arsitektur tiap layer
# inisialisasi bobot dan bias
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
bias = [0 for i in range(neuron[1])]
bias_2 = [0 for i in range(neuron[2])]
# bobot random dengan range -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2 * random.random() - 1
for i in range(neuron[1]):
    for j in range(neuron[2]):
        weight_2[i][j] = 2 * random.random() - 1

for e in range(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X):
        # Proses Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        # tandai class dengan nilai 1 [0,0,1]
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1
        # Cost function, Square Root Eror
        eror = 0
        for i in range(3):
            eror +=  0.5 * (target[i] - X_2[i]) ** 2
        cost_total += eror
        # Proses Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in range(neuron[2]):
            delta_2.append(-1 * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))
        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alfa * (delta_2[j] * X_1[i])
                bias_2[j] -= alfa * delta_2[j]
        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -=  alfa * (delta_1[j] * x[i])
                bias[j] -= alfa * delta_1[j]
    cost_total /= len(train_X)
    if(e % 100 == 0):
        print (cost_total) # Print cost untuk memantau training
#testing
res = matrix_mul_bias(test_X, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)
# mendapatkan prediction
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0])
# cetak prediction
print (preds)
# hitung accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print (acc / len(preds) * 100, "%")