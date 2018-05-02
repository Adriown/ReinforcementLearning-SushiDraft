#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:11:21 2018

@author: mead
"""

import os
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# I have my own function that will pull each year of the train data in a panda df
# You pass a directory and it returns a list of pandas dataframes
def getListDataFrames(path):
    # initialize an empty list
    csvs = list()
    # Get all the files in the directory
    all_files = os.listdir(path)
    # Iterate across each of the files in the Data folder
    for fileName in all_files:
        # Get the full path name
        fileToPull = path + fileName
        # Read in the data at that location
        filesOfInterest = pd.read_csv(fileToPull, index_col = False, low_memory = False)
        csvs.append(filesOfInterest)
    return csvs

# Here is the list of pds corresponding to the hyperparam search
listOfDfs = getListDataFrames('/Users/mead/Downloads/Results/')

HyperparamDfs = DataFrame()
for df in listOfDfs:
    HyperparamDfs = HyperparamDfs.append(df)

# Only look at the player we trained
HyperparamDfs = HyperparamDfs[HyperparamDfs['player'] == 0]
# Get rid of some weird extra column
HyperparamDfs = HyperparamDfs.drop('Unnamed: 0', axis = 1)
# Get only single values ( duplicates exist)
HyperparamDfs = HyperparamDfs.drop_duplicates()
# And sort by the best/worst performers
#HyperparamDfs = HyperparamDfs.sort_values('winPercent')
HyperparamDfs.sort_values('winPercent')[['method','nTrainIter','epsilon','alpha','lambda']].tail(45)
# ^ These are the top 45; they represent the best performers. All >30%

# PLOTTING
#test = HyperparamDfs.copy()
test = HyperparamDfs[HyperparamDfs['method'] != 'sarsa_lambda']

# Working on an accessible legend
add_lambda = test['lambda'].astype('str')
truth_ind = add_lambda == 'None'
add_lambda = ', lambda = ' + add_lambda
add_lambda[truth_ind] = ''
test['method'] = test['method'] + add_lambda
test = test.sort_values('method')
with sns.plotting_context('notebook', font_scale=1.3):
    g = sns.FacetGrid(test, col="epsilon",  row="alpha", hue = 'method', margin_titles = True, size = 2, aspect = 1.5)
    g = g.map(plt.axhline, y=.20, ls=":", c=".5")
    g = (g.map(plt.plot, "nTrainIter", "winPercent", marker = ".")
              .set(ylim = [.18, .36], xticks=[100, 250, 500, 1000])
              .add_legend())