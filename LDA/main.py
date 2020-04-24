import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import funcs

################# Configuration ##################
datasets = ['a1_d1','a1_d2']
thresholds = [-0.915, -4.4]
plot = True
#####################################################

for i in range(2):
    x = pd.read_csv("../Data/"+datasets[i]+".csv")
    y = x["y"].to_numpy()
    xpos, xneg, x = funcs.generate_two_classes(x)

    mean_diff = xpos.mean(0)-xneg.mean(0)

    cov_pos = funcs.covariance_matrix(xpos)
    cov_neg = funcs.covariance_matrix(xneg)

    pooled_covariance = cov_pos + cov_neg
    inverse_pooled_covariance = np.linalg.pinv(pooled_covariance)

    classifier = inverse_pooled_covariance.dot(mean_diff)
    xpos_transform = xpos.dot(classifier)
    xneg_transform = xneg.dot(classifier)
    
    predictions = funcs.classify(x, classifier, thresholds[i], datasets[i])

    print('Results for '+datasets[i])
    funcs.evaluate(predictions, y,datasets[i])
    print('\n\n\n')
    if plot == True:
        funcs.plot_normal(xpos_transform, xneg_transform, datasets[i])
    
