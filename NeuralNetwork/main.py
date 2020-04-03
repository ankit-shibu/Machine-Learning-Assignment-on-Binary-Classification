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
 
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

xTrain = xTrain.to_numpy() 
yTrain = yTrain.to_numpy()

xTest = xTest.to_numpy() 
yTest = yTest.to_numpy()  

################   INPUT PARAMS   ###################
no_of_layers = 3
no_of_nodes = [xTrain.shape[1],2,1]
data_scaler = 'standardization'
activations = ['sigmoid','sigmoid']
no_of_iters = 3
size_of_batch = 1000
learning_rate = 0.01
#####################################################

if data_scaler == 'standardization':
    xTrain = preprocessing.scale(xTrain)
else:
    xTrain = preprocessing.MinMaxScaler.transform(xTrain)

##Creating the initial population.
weights = []
bias = []

for i in numpy.arange(0, no_of_layers-1):
    layera = no_of_nodes[i]
    layerb = no_of_nodes[i+1]
    weight_layer = numpy.random.uniform(low=-0.5, high=0.5, 
                                                 size=(layera, layerb))
    bias_layer = numpy.random.randn(1, layerb)
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
                dweights, dbias = NN.backpropagation(y_batch, plain_values, activated_values, weights, activations)
                weights = [w+learning_rate*dwt for w,dwt in  zip(weights, dweights)]
                bias = [w+learning_rate*dbs for w,dbs in  zip(bias, dbias)]
                print('loss = '+str(NN.loss_function(activated_values[-1],y_batch)))

plain_values, activated_values = NN.forwardpropogation(xTrain, weights,activations,bias)
predictions = activated_values[-1]
pred = numpy.copy(predictions)
pred[pred>=0.5] = 1
pred[pred<0.5] = 0
print(pred)
    







