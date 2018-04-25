#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alizaidi
"""
from SushiDraft import *

final_df_qm = DataFrame()
final_df_sl = DataFrame()

for i in np.arange(0.1, 0.5, 0.1):
    for j in np.arange(0.5, 0.6, 0.1):
        qLearnSpace, qLearnWinPercents = qLearning(possStateActions, epsilon = i, alpha = j, 
                                                   measureWinPoints = [100],
                                                   numIterations = [50])
        
        monteCarloESSpace, monteCarloESWinPercents = monteCarloES(possStateActions, epsilon = i, alpha = j, 
                                                   measureWinPoints = [100],
                                                   numIterations = [50])
        
        qLearnWinPercents["epsilon"] = i
        qLearnWinPercents["alpha"] = j
        qLearnWinPercents["lambda"] = "None"
        
        monteCarloESWinPercents["epsilon"] = i
        monteCarloESWinPercents["alpha"] = j
        monteCarloESWinPercents["lambda"] = "None"
        
        final_df_qm = final_df_qm.append(qLearnWinPercents)
        final_df_qm = final_df_qm.append(monteCarloESWinPercents)
        
        
for i in np.arange(0.1, 0.5, 0.1):
    for j in np.arange(0.5, 0.6, 0.1):
        for k in np.arange(0, 0.5, 0.1):
            sarsaLambdaSpace, sarsaLambdaWinPercents = sarsa_lambda(possStateActions, epsilon = i, alpha = j, 
                                                                    lambda_ = k,
                                                                    measureWinPoints = [100],
                                                                    numIterations = [50])
            sarsaLambdaWinPercents["epsilon"] = i
            sarsaLambdaWinPercents["alpha"] = j
            sarsaLambdaWinPercents["lambda"] = k
            
            final_df_sl = final_df_sl.append(sarsaLambdaWinPercents)