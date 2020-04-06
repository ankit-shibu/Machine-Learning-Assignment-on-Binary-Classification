import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

def generate_two_classes(x):
    xpos = x[x["y"]==1]
    columns = xpos.columns.tolist()
    cols_to_use_input = columns[:len(columns)-1]
    xpos = xpos[cols_to_use_input]
        
    xneg = x[x["y"]==0]
    columns = xneg.columns.tolist()
    cols_to_use_input = columns[:len(columns)-1]
    xneg = xneg[cols_to_use_input]
    
    x = x[cols_to_use_input]
        
    return xpos, xneg, x

def covariance_matrix(x):
    n = np.shape(x)[0]
    covariance_matrix = (1 / (n-1)) * (x - x.mean(0)).T.dot(x - x.mean(0))
    return np.array(covariance_matrix, dtype=float)

def plot_normal(x1, x2):
    x = np.linspace(-20, 20, 100)
    
    ypos = scipy.stats.norm.pdf(x, x1.mean(0), x1.std(0))
    yneg = scipy.stats.norm.pdf(x, x2.mean(0), x2.std(0))
    
    plt.plot(x, ypos, color='red')
    plt.plot(x, yneg, color='blue')

def classify(x, classifier, thres):
    ylabel = []
    x = x.dot(classifier)
    for i in range(len(x)):
        if x[i] > thres:
            ylabel.append(1)
        else:
            ylabel.append(0)
    
    return ylabel

def evaluate(ylabel, ytest):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(ylabel)):
        if ytest[i] == ylabel[i] and ytest[i] == 1 :
            tp = tp+1
        elif ytest[i] == ylabel[i] and ytest[i] == 0:
            tn = tn+1
        if ytest[i] != ylabel[i] and ytest[i] == 1 :
            fn = fn + 1
        elif ytest[i] != ylabel[i] and ytest[i] == 0:
            fp = fp + 1

    print("True Positives = "+str(tp))
    print("False Positives = "+str(fp))
    print("True Negatives = "+str(tn))
    print("False Negatives = "+str(fn))

    Precision = tp/(tp+fp)
    Accuracy = (tp+tn)/(tp+tn+fp+fn)
    Recall = tp/(tp+fn)
    
    F1_score = 2*Precision*Recall/(Precision+Recall)
    FPR = fp/(fp+tn)
    Specificity = 1 - FPR
    print("Precision = "+str(Precision))
    print("Accuracy = "+str(Accuracy))
    print("Recall = "+str(Recall))

    print("F1 Score = "+str(F1_score))
    print("FPR = "+str(FPR))
    print("Specificity = "+str(Specificity))
    
    
    
