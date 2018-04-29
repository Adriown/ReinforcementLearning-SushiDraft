#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:43:13 2018

@author: Jo
"""
import pandas as pd
import pickle #for saving and opening the python object

'''
NO NEED TO RUN [Begin] --------------
The following chunk of code is building the new state_actions space.
I saved the python object into a new file. 
'''


#run SushiDraft to get **possStateActions** (the old version, to be updated) [37379 rows]
#run NewStateSpace_boolean.py to get **possStates** (the updated one) [296595 rows]

import ast #for convertion from 'list' to [list]

possStates_df=DataFrame.from_records(possStates) 
new_possStates=DataFrame.copy(possStates_df)
new_possStates['state']=new_possStates[[0,1]].values.tolist() #combine the  hand and column like the possStateActions
new_possStates['state']=new_possStates['state'].apply(str) #convert to string like the possStateActions
new_possStatesActions=pd.merge(new_possStates, possStateActions, on='state')  #[1586152 rows] 

new_possStatesActions['state']=new_possStatesActions[[0,1,2]].values.tolist()

possStatesActions=new_possStatesActions[['state','action']]


# Saving the objects:
possStatesActions.to_pickle('/Users/Pan/Google Drive/Data Science/Reinforcement Learning/ReinforcementLearning-SushiDraft/possStateActions.pkl')


'''
NO NEED TO RUN [End] -------------
'''

# Getting back the possStateActions object:
f=open('/Users/Pan/Google Drive/Data Science/Reinforcement Learning/ReinforcementLearning-SushiDraft/possStateActions.pkl','rb')  
possStateActions = pickle.load(f)
f.close()

possStateActions['Q'] = 0
    
# NEW FUNCTION to get 
def get_others_boolean(played_cards):
    other_bool = []
    
    played=currentState(played_cards[0])
    
    others=[]
    for i in range(1,len(played_cards)):
        others=others.append(currentState(played_cards[i]))
    
    # max other
    other=map(max,zip(played_cards[1:len(played_cards)]))
    
    for x, y in zip(played, other):
        if y < x:
            other_bool.append(-1)
        elif y == x:
            other_bool.append(0)
        else:
            other_bool.append(1)
    
    return other_bool

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
            curr_played_cards=dummy.played_cards
            currState = [currentState(dummy.hand_cards[0]),
                         currentState(curr_played_cards[0]),get_others_boolean(curr_played_cards)]
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
            curr_played_cards=dummy.played_cards
            nextState = [currentState(dummy.hand_cards[0]),
                         currentState(curr_played_cards[0]),get_others_boolean(curr_played_cards)]
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