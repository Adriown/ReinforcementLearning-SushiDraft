#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:44:18 2018

@author: mead
"""

#from SushiDraft import *


np.random.seed(12345)

# We get trainings from qLearning using random policy to both train and evaluate
qLearnSpace, qLearnWinPercents = qLearning(possStateActions, measureWinPoints = [100,1000, 2000],
                                           numIterations = [400, 400, 400])


monteCarloESSpace, monteCarloESWinPercents = monteCarloES(possStateActions, measureWinPoints = [100,1000, 2000],
                                           numIterations = [400, 400, 400])


sarsaLambdaSpace, sarsaLambdaWinPercents = sarsa_lambda(possStateActions, measureWinPoints = [100,1000, 2000],
                                           numIterations = [400, 400, 400])

print(qLearnWinPercents[['method', 'nTrainIter', 'nTrials', 'winPercent']].loc[0])
print(monteCarloESWinPercents[['method', 'nTrainIter', 'nTrials', 'winPercent']].loc[0])
print(sarsaLambdaWinPercents[['method', 'nTrainIter', 'nTrials', 'winPercent']].loc[0])

evaluatePolicy(2000, 500, sarsaLambdaSpace, 5, 'sarsaLambdaWinPercents', policySpace = None)


sarsaLambdaSpace2, sarsaLambdaWinPercents2 = sarsa_lambda(possStateActions, measureWinPoints = [100,1000, 2000],
                                           numIterations = [400, 400, 400], trainPolicySpace = sarsaLambdaSpace)


a = evaluatePolicy(2000, 500, sarsaLambdaSpace2, 5, 'sarsaLambdaWinPercents', policySpace = None)
b = evaluatePolicy(2000, 500, sarsaLambdaSpace2, 5, 'sarsaLambdaWinPercents', policySpace = sarsaLambdaSpace)
c = evaluatePolicy(2000, 500, sarsaLambdaSpace, 5, 'sarsaLambdaWinPercents', policySpace = sarsaLambdaSpace2)

sarsaLambdaSpace3, sarsaLambdaWinPercents3 = sarsa_lambda(sarsaLambdaSpace, measureWinPoints = [100,1000, 2000],
                                           numIterations = [400, 400, 400], trainPolicySpace = sarsaLambdaSpace)

d = evaluatePolicy(2000, 1000, sarsaLambdaSpace, 5, 'sarsaLambdaWinPercents', policySpace = sarsaLambdaSpace3)

# Does it ever get better?
sarsaLambdaSpace4, sarsaLambdaWinPercents4 = sarsa_lambda(sarsaLambdaSpace, measureWinPoints = [100,1000, 2000],
                                           numIterations = [400, 400, 400], evalPolicySpace = sarsaLambdaSpace)
sarsaLambdaSpace4
evaluatePolicy(2000, 500, sarsaLambdaSpace, 5, 'sarsaLambdaWinPercents', policySpace = sarsaLambdaSpace)