import numpy
import NN
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

x = pd.read_csv("../Data/housepricedata.csv")
columns = x.columns.tolist()
cols_to_use_input = columns[:len(columns)-1]
cols_to_use_output = columns[len(columns)-1:]

x = pd.read_csv("../Data/housepricedata.csv", usecols = cols_to_use_input)
y = pd.read_csv("../Data/housepricedata.csv", usecols = cols_to_use_output)

x = x.to_numpy()
y = y.to_numpy()
################   INPUT PARAMS   ###################
no_of_layers = 3
no_of_nodes = [10,5,1]
data_scaler = 'standardizaion'
activations = ['relu','sigmoid']
no_of_iters = 100000
size_of_batch = 1168
learning_rate = 0.0001
#####################################################

if data_scaler == 'standardization':
    x = preprocessing.scale(x)
else:
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

##Creating the initial population.
weights = []
bias = []

for i in numpy.arange(0, no_of_layers-1):
    #numpy.random.seed(0)
    layera = no_of_nodes[i]
    layerb = no_of_nodes[i+1]
    weight_layer = numpy.random.randn(layerb,layera)/numpy.sqrt(layera)
    bias_layer = numpy.zeros((layerb, 1))
    weights.append(weight_layer)
    bias.append(bias_layer)
##################################################


for itno in range(no_of_iters): 
            i=0
            while(i<len(yTrain)):
                x_batch = xTrain[i:i+size_of_batch]
                y_batch = yTrain[i:i+size_of_batch]
                i = i+size_of_batch
                plain_values, activated_values = NN.forwardpropogation(x_batch, weights,activations,bias)
                if itno%1000 == 0 or itno==0:
                    print('loss = '+str(NN.loss_function(activated_values[-1],y_batch)))
                delta = NN.backpropagation(y_batch, plain_values, activated_values, weights, activations,no_of_layers)
                for layer in range(no_of_layers-1):
                    weights[layer] = weights[layer] - learning_rate * delta["dweights"+str(layer)]
                    bias[layer] = bias[layer] - learning_rate * delta["dbias"+str(layer)]            
                
NN.evaluate(xTrain, yTrain, weights, activations, bias)
NN.evaluate(xTest, yTest, weights, activations, bias)







