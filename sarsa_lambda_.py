#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 21:01:06 2018

@author: Pan
"""

import numpy as np

def sarsa_lambda(qStateActionSpace,
                 measureWinPoints,numIterations,
                 alpha = .5, gamma = 1, lambda_ = .1, epsilon=0.9):
    """
    This function implements backward view of Sarsa(lambda). 
        
    Notes on inputs:    
    -transition: function. It takes current state s and action a as parameters 
                and returns next state s', immediate reward R, and a boolean 
                variable indicating whether s' is a terminal state. 
                (See windy_setup as an example)
    -epsilon: exploration rate as in epsilon-greedy policy. 
               IS IT CORRECT TO USE 0.9?
    -measureWinPoints: an ndarray that lists the number of games to be played before evaluating
                        the win rate
    -numIterations:the number of times we want to simulate using the optimal policy
                     to find the win percentage
    
    """    
    
    win_percents = DataFrame() # Track the win percentages across players (draws count as wins)
    
    for i in range(1, max(measureWinPoints) + 1): # for every episode
        E=qStateActionSpace.copy()
        totalReward = 0
        dummy = SushiDraft(1, 5, score_tokens, deck, 0) # random initialization of the game
        isPlaying = 1
        
        
        while(isPlaying): # while not terminate, play a round
            currState = [currentState(dummy.hand_cards[0]),currentState(dummy.played_cards[0])]
            # Selecting the possible actions corresponding to this current state
            possActions = qStateActionSpace[qStateActionSpace['state'] == str(currState)]
            # Epsilon-greedy implementation
            greedy_prob = 1 - epsilon
            # Now decide which action to take
            if np.random.random() < greedy_prob:
                # Take the greedy action
                muActionIndex = possActions.sample(len(possActions))['Q'].idxmax()
                piActionIndex = muActionIndex.copy()
            else:
                # Take a random action
                muActionIndex = possActions.sample(1)['Q'].idxmax()
                piActionIndex = possActions.sample(len(possActions))['Q'].idxmax()
            
            # Now record what our character is going to do
            play_card, keep_card, is_wildcard = possActions.loc[muActionIndex]['action']
            
            # Figure out what the competition is going to do
            play_cards, keep_cards, is_wildcards = randomMoves(dummy.hand_cards, dummy.played_cards, range(1, dummy.num_players))
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
            nextState = [currentState(dummy.hand_cards[0]),currentState(dummy.played_cards[0])]
            # Selecting the possible actions corresponding to this current state
            possNextActions = qStateActionSpace[qStateActionSpace['state'] == str(nextState)]
            # Check if we finished the round just now
            if dummy.num_cards_played != 0:
                piNextActionIndex = possNextActions.sample(len(possNextActions))['Q'].idxmax()
                # PERFORM THE Q-update
                delta=immedReward+gamma*qStateActionSpace.loc[piNextActionIndex, 'Q']- qStateActionSpace.loc[muActionIndex, 'Q']
                E.loc[muActionIndex, 'Q']+=1
                for s in range(len(qStateActionSpace)):   
                    qStateActionSpace.loc[s, 'Q'] += alpha * delta * E.loc[s, 'Q']
                    E.loc[s,'Q']=gamma*lambda_*E.loc[s,'Q']
            else:
    #            print("End of the round")
    #            print("Immediate Reward: " + str(immedReward))
                # This is the case where the round was finished
                # Think about if we want to keep it this way. We're basically saying the terminal state has value 0, which is probably reasonable
                delta=immedReward+gamma*0- qStateActionSpace.loc[muActionIndex, 'Q']
                E.loc[muActionIndex, 'Q']+=1
                for s in range(len(qStateActionSpace)):   
                    qStateActionSpace.loc[s, 'Q'] += alpha * delta * E.loc[s, 'Q']
                    E.loc[s,'Q']=gamma*lambda_*E.loc[s,'Q']
                
            
        # THIS PORTION IS ABOUT GETTING THE PERFORMANCE OF OUR OPTIMAL POLICY
        if np.any(i == measureWinPoints): # Run this once we hit a certain number of game iterations run
            # Get the trials we want to accomplish to check our win percentage
            num_trials = numIterations[np.where(i == measureWinPoints)[0]][0]
            print("Evaluating the win percentage of our trained agent after " + str(i) + \
                  " iterations of the game. Going to perform " + str(num_trials) + " trials.")
            # Initialize some counts of to keep track of winners
            win_counts = Series([0] * dummy.num_players, range(dummy.num_players))
            # We can use these next two to figure out what percentage of the state-space we actually visited across out trials
            no_value = 0 
            total_values = 0
            # Run the optimal policy across num_trials number of games
            for j in range(num_trials):
    #                if j % 10 == 0:
    #                    print(i)
                totalReward = 0
                dummy = SushiDraft(1, 5, score_tokens, deck, 0) # random initialization of the game
                isPlaying = 1
                while(isPlaying):
                    total_values += 1
                    currState = [currentState(dummy.hand_cards[0]),
                                 currentState(dummy.played_cards[0])]
                    # Selecting the possible actions corresponding to this current state
                    possActions = qStateActionSpace[qStateActionSpace['state'] == str(currState)]
                    
                    # Figure out what proportion of states haven't been visited
                    if possActions.sample(len(possActions))['Q'].max() == 0:
                        no_value += 1
                    
                    # Now decide which action to take -- follow optimal
                    piActionIndex = possActions.sample(len(possActions))['Q'].idxmax()
                    
                    # Now record what our character is going to do
                    play_card, keep_card, is_wildcard = possActions.loc[piActionIndex]['action']
                    
                    # Figure out what the competition is going to do
                    play_cards, keep_cards, is_wildcards = randomMoves(dummy.hand_cards, dummy.played_cards, range(1, dummy.num_players))
                    play_cards.insert(0, play_card)
                    keep_cards.insert(0, keep_card)
                    is_wildcards.insert(0, is_wildcard)
                        
                    # Take a turn of the game
                    isPlaying = dummy.takeTurn(play_cards, keep_cards, is_wildcards)
                # Figure out who won
                win_counts += getWinner(dummy.player_tokens)
            # Go through and format the DataFrame() the way we want
            win_percent = DataFrame(win_counts, columns = ['nWins'])
            win_percent['nTrials'] = num_trials
            win_percent['player'] = range(dummy.num_players)
            win_percent['nTrainIter'] = i
            win_percent['winPercent'] = win_percent['nWins'] / win_percent['nTrials']
            # Now append the new percentages
            win_percents = win_percents.append(win_percent)
    # Append a method for the sake of remembering what we did
    win_percents['method'] = 'sarsa_lambda'
    win_percents = win_percents[['method', 'player','nTrainIter', 'nTrials', 'nWins', 'winPercent']]
    qStateActionSpace['method'] = 'q-sarsa_lambda'
    qStateActionSpace = qStateActionSpace[['method', 'state', 'action', 'Q']]
    # Return the q-state action values and our optimal policy win rates
    return (qStateActionSpace, win_percents)