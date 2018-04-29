#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:43:13 2018

@author: Pan
"""
import pandas as pd
import pickle #for saving and opening the python object
#Run functions in SushiDraft.py (all above line 362)


'''
NO NEED TO RUN [Begin]
The following chunk of code is building the new state_actions space.
I saved the python object into a new file. 
'''
import csv 
import ast #for convertion from pd.df to list of list
filename="/Users/Pan/Google Drive/Data Science/Reinforcement Learning/ReinforcementLearning-SushiDraft/statespace_boolean.csv"
possStates_df=pd.read_csv(filename)
possStates_df=possStates_df.drop(['Unnamed: 0'],axis=1)
        #0: Hand states
        #1: Played States
        #2: Other States

# The original possStates is a list of list. 
# The following part is trying to convert pd to list of list. (sorry that it is kind of slow)
possStates=list()
for i in range(len(possStates_df)):
    theRow=list()
    for j in range(3):      
        theRow.append(ast.literal_eval(possStates_df.iloc[i][j]))
    possStates.append(theRow)                  

# THIS PART IS HUGE. I'm constructing a comprehensive state-action space to hold. sorry, it is very very slow.
# Used the function in SushiDraft.py
possStateActions = DataFrame()
possStatesLen=len(possStates)
for i in range(possStatesLen): # Run through each of the states we've identified
    if i%1000==0:
        print ("{0:.0f}%".format(i/possStatesLen * 100))
        break
    possStateAction = DataFrame()
    # And now identify every single possible action for that state
    possStateAction['action'] = possibleActions(possStates[i][0], possStates[i][1])
    # Note that in order to get this to work, I needed to make the state a string
    possStateActions['state'] = str(possStates[i])
    possStateActions = possStateActions.append(possStateAction)
possStateActions = possStateActions.reset_index()
possStateActions = possStateActions.drop('index', axis = 1)
possStateActions['Q'] = 0
possStateActions = possStateActions[['state','action','Q']]

# Saving the objects:
# https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
f=open('possStateActions.pkl', 'wb')  
pickle.dump(possStateActions, f)
f.close()

'''
NO NEED TO RUN [End]
'''

# Getting back the possStateActions object:
f=open('possStateActions.pkl','rb')  
possStateActions = pickle.load(f)
f.close()
    

def qLearning(possStateActions, epsilon = .9, alpha = .5, gamma = 1, 
              measureWinPoints = np.asarray([10, 20]), numIterations = np.asarray([20, 30]),
              numPlayers = 5, score_tokens = score_tokens, deck = deck, 
              trainPolicySpace = None, evalPolicySpace= None):
    """
    THIS IS WHERE THE HEAVY LIFTING IS HAPPENING.
    
    The goal here is to come up with a generalizable framework for policy control 
    that can be applied/transferred across multiple state-action spaces and with 
    the ability to, mid-control, evaluate the performance of our optimal policy up
    to that point.
    
    As input, pass a qStateActionSpace DataFrame(), hyperparameters for learning (epsilon, alpha, gamma),
    measureWinPoints (an ndarray that lists the number of games to be played before evaluating
    the win rate), and numIterations (the number of times we want to simulate using the optimal policy
    to find the win percentage)
    
    As time goes on this function may need upgrades to allow us to play against more than just the random
    policy. For instance, we may want to be able to have a certain policy being trained against a 
    different policy.
    
    NOTE! Currently the reward is just equal to the amount won in a round. This means that 
    learning is quite slow. We may want to brainstorm a better way to do it and introduce
    some sort of negative rewards.
    
    This function returns an updated qStateActionSpace and DataFrames with information about
    the win rates of our optimal policy at different measureWinPoints
    """
    qStateActionSpace = possStateActions.copy()
    measureWinPoints = np.asarray(measureWinPoints)
    numIterations = np.asarray(numIterations)
    win_percents = DataFrame() # Track the win percentages across players (draws count as wins)
    for i in range(1, max(measureWinPoints) + 1): # Perform the algorithm as many times as we want
        totalReward = 0
        dummy = SushiDraft(1, numPlayers, score_tokens, deck, 0) # random initialization of the game
        isPlaying = 1
        while(isPlaying): # Run through one full game, peforming control as we go
            curr_played_cards
            currState = [currentState(dummy.hand_cards[0]),
                         currentState(dummy.played_cards[0])]
            # Selecting the possible actions corresponding to this current state
            possActions = qStateActionSpace[qStateActionSpace['state'] == str(currState)]
            # Epsilon-greedy implementation
            greedy_prob = 1 - epsilon
            # Now decide which action to take
            if np.random.random() < greedy_prob:
                # Take the greedy action
                muActionIndex = possActions.sample(len(possActions))['Q'].idxmax()
#                piActionIndex = muActionIndex.copy()
            else:
                # Take a random action
                muActionIndex = possActions.sample(1)['Q'].idxmax()
#                piActionIndex = possActions.sample(len(possActions))['Q'].idxmax()
            
            # Now record what our character is going to do
            play_card, keep_card, is_wildcard = possActions.loc[muActionIndex]['action']
            
            # Figure out what the competition is going to do
            if trainPolicySpace is None: # Use the random agent to play
                play_cards, keep_cards, is_wildcards = randomMoves(dummy.hand_cards, dummy.played_cards, range(1, dummy.num_players))
            else: # Use a trained agent to play
                play_cards, keep_cards, is_wildcards = policyMoves(dummy.hand_cards, dummy.played_cards, trainPolicySpace, range(1, dummy.num_players))
            play_cards.insert(0, play_card)
            keep_cards.insert(0, keep_card)
            is_wildcards.insert(0, is_wildcard)
                
            # Take a turn of the game
            isPlaying = dummy.takeTurn(play_cards, keep_cards, is_wildcards)
            
            # REWARDS ARE HERE
            # Check if the round ended. If so, reward will be equal to the value of the accrued score tokens
            if dummy.num_cards_played == 0: # This occurs when we hit the end of a round and score tokens are passed out
                immedReward = dummy.player_tokens[0] - totalReward
                totalReward = dummy.player_tokens[0].copy()
            else:
                immedReward = 0
            
            # Figure out the Q-value of the next state
            nextState = [currentState(dummy.hand_cards[0]),
                         currentState(dummy.played_cards[0])]
            # Selecting the possible actions corresponding to this current state
            possNextActions = qStateActionSpace[qStateActionSpace['state'] == str(nextState)]
            # Check if we finished the round just now
            if dummy.num_cards_played != 0:
                piNextActionIndex = possNextActions.sample(len(possNextActions))['Q'].idxmax()
                # PERFORM THE Q-update
                qStateActionSpace.loc[muActionIndex, 'Q'] += alpha * (immedReward + gamma * qStateActionSpace.loc[piNextActionIndex, 'Q'] - qStateActionSpace.loc[muActionIndex, 'Q'])
            else:
    #            print("End of the round")
    #            print("Immediate Reward: " + str(immedReward))
                # This is the case where the round was finished
                # Think about if we want to keep it this way. We're basically saying the terminal state has value 0, which is probably reasonable
                qStateActionSpace.loc[muActionIndex, 'Q'] += alpha * (immedReward + gamma * 0 - qStateActionSpace.loc[muActionIndex, 'Q'])
            
        # THIS PORTION IS ABOUT GETTING THE PERFORMANCE OF OUR OPTIMAL POLICY
        if np.any(i == measureWinPoints): # Run this once we hit a certain number of game iterations run
            # Get the trials we want to accomplish to check our win percentage
            num_trials = numIterations[np.where(i == measureWinPoints)[0]][0]
            print("Evaluating the win percentage of our trained agent after " + str(i) + \
                  " iterations of the game. Going to perform " + str(num_trials) + " trials.")
            win_percent = evaluatePolicy(i, num_trials, qStateActionSpace, 
                                         numPlayers, 'q-learning', evalPolicySpace)
            # Now append the new percentages
            win_percents = win_percents.append(win_percent)
    qStateActionSpace['method'] = 'q-learning'
    qStateActionSpace = qStateActionSpace[['method', 'state', 'action', 'Q']]
    # Return the q-state action values and our optimal policy win rates
    return (qStateActionSpace, win_percents)