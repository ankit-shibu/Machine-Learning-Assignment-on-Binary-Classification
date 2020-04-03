import numpy
import math

def sigmoid(inpt):
    return 1.0/(1.0+numpy.exp(-1*inpt))

def linear(inpt):
    return inpt

def tanh(inpt):
    return numpy.tanh(inpt)

def relu(inpt):
    result = inpt
    result[inpt<0] = 0
    return result

def derivative_sigmoid(inpt):
    sig = sigmoid(inpt)
    return sig * (1 - sig)

def derivative_linear(inpt):
    return 1

def derivative_tanh(inpt):
    t = tanh(inpt)
    return (1 - (t**2))

def derivative_relu(inpt):
    inp = numpy.copy(inpt)
    inp[inp<0] = 0
    return inp

def forwardpropogation(x, weights, activations, bias):
        r1 = numpy.copy(x)
        plain_values = []
        activated_values = [r1]
        for i in range(len(weights)):
            r1 = numpy.matmul(r1, weights[i])
            plain_values.append(r1 + bias[i])
            if activations[i] == "relu":
                r1 = relu(r1)
            elif activations[i] == "sigmoid":
                r1 = sigmoid(r1)
            elif activations[i] == "linear":
                r1 = linear(r1)
            elif activations[i] == "tanh":
                r1 = tanh(r1)
            activated_values.append(r1)
        return (plain_values, activated_values)

def backpropagation1(y, plain_values, activated_values, weights, activations):
        dweights = []  
        dbias = []
        delta = [None] * len(weights)  

        if activations[-1] == "relu":
                delta[-1] = ((y-activated_values[-1])*derivative_relu(plain_values[-1]))
        elif activations[-1] == "sigmoid":
                delta[-1] = ((y-activated_values[-1])*derivative_sigmoid(plain_values[-1]))
        elif activations[-1] == "linear":
                delta[-1] = ((y-activated_values[-1])*derivative_linear(plain_values[-1]))
        elif activations[-1] == "tanh":
                delta[-1] = ((y-activated_values[-1])*derivative_tanh(plain_values[-1]))
                
        for i in reversed(range(len(delta)-1)):
            if activations[i] == "relu":
                    delta[i] = delta[i+1]*derivative_relu(plain_values[i])
            elif activations[i] == "sigmoid":
                    delta[i] = delta[i+1]*derivative_sigmoid(plain_values[i])
            elif activations[i] == "linear":
                    delta[i] = delta[i+1]*derivative_linear(plain_values[i])
            elif activations[i] == "tanh":
                    delta[i] = delta[i+1]*derivative_tanh(plain_values[i])
        print(delta[i].shape)           
        batch_size = y.shape[1]
        dbias = [d.dot(numpy.ones((batch_size,1)))/float(batch_size) for d in delta]
        dweights = [d.dot(activated_values[i].T)/float(batch_size) for i,d in enumerate(delta)]

        return dweights, dbias

def backpropagation(y, plain_values, activated_values, weights, activations):
         dweights = []  
         dbias = []
         rweights = []
         rbias = []
         
         dAL = - (numpy.divide(y, activated_values[-1]) - numpy.divide(1 - y, 1 - activated_values[-1]))
         dA_prev = dAL
         for l in reversed(range(len(weights))):
                dA_curr = dA_prev
                
                W_curr = weights[l]
                Z_curr = plain_values[l]
                A_prev = activated_values[l]
        
                dA_prev, dW_curr, db_curr = linear_activation_backward(dA_curr, Z_curr, A_prev, W_curr, activations[l])
        
                rweights.append(dW_curr)
                rbias.append(db_curr)
         
         for l in reversed(range(len(weights))):
             dweights.append(rweights[l])
             dbias.append(rbias[l])

         dweights = numpy.array(dweights)
         dbias = numpy.array(dbias)
         return dweights, dbias
        
def linear_activation_backward(dA, Z, A_prev, W, activation):
            if activation == "relu":
                dZ = dA * derivative_relu(Z)
                dA_prev, dW, db = linear_backward(dZ, A_prev, W)
            elif activation == "sigmoid":
                dZ = derivative_relu(Z)
                dA_prev, dW, db = linear_backward(dZ, A_prev, W)
            elif activation == "linear":
                dZ = dA * derivative_linear(dA, Z)
                dA_prev, dW, db = linear_backward(dZ, A_prev, W)
            elif activation == "tanh":
                dZ = dA * derivative_tanh(dA, Z)
                dA_prev, dW, db = linear_backward(dZ, A_prev, W)
        
            return dA_prev, dW, db
        
def linear_backward(dZ, A_prev, W):
            m = A_prev.shape[1]     
            dW = numpy.dot(A_prev.T, dZ) / m
            db = numpy.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = numpy.dot(dZ, W.T)
        
            return dA_prev, dW, db
    
def loss_function(prediction, actual):
   size = actual.shape[1]
   loss = numpy.multiply(numpy.log(prediction),actual) + numpy.multiply(1 - actual, numpy.log(1 - prediction))
   return - numpy.sum(loss) / size

    



