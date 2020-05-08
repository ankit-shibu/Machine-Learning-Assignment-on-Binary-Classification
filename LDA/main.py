import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import funcs

################# Configuration ##################
datasets = ['a1_d1','a1_d2']
thresholds = [-0.915, -4.4]
plot = False
#####################################################

for i in range(2):
    x = pd.read_csv("../Data/"+datasets[i]+".csv")
    xtrain = x[0:int(0.8*len(x))]
    xtest = x[int(0.8*len(x)):len(x)]
    ytrain = xtrain["y"].to_numpy()
    ytest = xtest["y"].to_numpy()
    xpos, xneg, x, xtest = funcs.generate_two_classes(xtrain, xtest)

    mean_diff = xpos.mean(0)-xneg.mean(0)

    cov_pos = funcs.covariance_matrix(xpos)
    cov_neg = funcs.covariance_matrix(xneg)

    pooled_covariance = cov_pos + cov_neg
    inverse_pooled_covariance = np.linalg.pinv(pooled_covariance)

    classifier = inverse_pooled_covariance.dot(mean_diff)
    xpos_transform = xpos.dot(classifier)
    xneg_transform = xneg.dot(classifier)
    
    predictions = funcs.classify(xtest, classifier, thresholds[i], datasets[i])

    print('Results for '+datasets[i])
    funcs.evaluate(predictions, ytest,datasets[i])
    print('\n\n\n')
    if plot == True:
        funcs.plot_normal(xpos_transform, xneg_transform, datasets[i])
    
