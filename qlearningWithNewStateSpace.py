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
'''
Adrian's new definition of the class  ----------
    ends at line 194
'''

class SushiDraft:
    """
    The SushiDraft class embodies an instance of the game. It initializes a configuration
    of the game based on the parameters passed to it. By default, the behavior is to 
    initialize a random game at the very beginning of play.
    
    The primary way of interacting with the SushiDraft instance from outside of the class
    is with the takeTurn function. takeTurn() is passed cards to play for each player, cards to
    save for each player, and if any of the cards played are being done as a wildcard
    """
    def __init__(self, num_round, num_players, score_tokens_avail, deck, num_cards_played, hand_cards = [], played_cards = [], player_tokens = Series(), one_player_hands = [], suppressPrint = True):  # constructor
#        fields: num_round, num_players, score_tokens_avail (a Series), deck (a list), num_cards_played 
#        (from 0 to 4 when initializing), hand_cards (a list of ndarrays), played_cards 
#        (a list of ndarrays), and player_tokens (a Series)
        # Initialization function
        self.num_round = num_round
        self.num_players = num_players
        self.score_tokens_avail = score_tokens_avail
        self.deck = deck
        self.num_cards_played = num_cards_played
        self.one_player_hands = one_player_hands
        self.suppressPrint = suppressPrint
        if len(hand_cards) == 0:
            # This is basically the case where you shuffle the deck
#            np.random.seed(1)
            if len(one_player_hands) != 0: # This is the case where we go in and explicity give the first player a specific hand
                new_deck = deck.copy()
                [new_deck.remove(card) for card in one_player_hands[0]]
                self.hand_cards = np.array_split(np.random.choice(new_deck, 6 * (num_players - 1), False), num_players - 1)
                self.hand_cards.insert(0, one_player_hands[0])
            else: # Everyone gets a random hand
                self.hand_cards = np.array_split(np.random.choice(deck, 6 * num_players, False), num_players)
        else:
            # If we want to initialize a non-random game with specific hands
            self.hand_cards = hand_cards
        if len(played_cards) == 0:
            # No cards played yet in default case
            self.played_cards = [np.empty(0, dtype='int') for i in range(num_players)]
        else:
            self.played_cards = played_cards
        if len(player_tokens) == 0:
            # This is just going to initialize everyone to having no points yet
            self.player_tokens = Series([0 for i in range(num_players)], 
                                        [i for i in range(num_players)])
        else:
            self.player_tokens = player_tokens
            
    def takeTurn(self, play_cards, save_cards = [], is_wild = [0, 0, 0, 0, 0]): # go through a move
        # fields: play_cards (a list of the cards played in order),
        #         save_cards (a list of the cards saved in order), and
        #         is_wild (a list saying if a card being played is a wildcard (2)) <- only referring to play_cards
        # Track how many cards have been played
        self.num_cards_played += 1
        # Add the play_cards to the played_cards for each player
        self.played_cards = [np.append(self.played_cards[i], play_cards[i]) for i in range(len(play_cards))] 
        if self.num_cards_played == 5:  # All cards have been played for the round
            # GO THROUGH THE MOTION OF EVALUATING THE PLAYERS AND HANDING OUT TOKENS
            val_counts = [np.unique(player, return_counts=True) for player in self.played_cards]
            for group in np.unique(list(self.score_tokens_avail.index)): # UPDATE SCORES
                if group == 2:
                    # THIS IS FOR THE PLAYER WITH THE MOST DIVERSITY
                    group_counts = [len(player[0]) for player in val_counts]
                    best_player = SushiDraft.winningPlayer(group_counts)
                    if best_player is None:
                        continue
                    self.passScoreToken(best_player, group)
                else:
                    # THIS IS FOR ALL OTHER CASES; WE'RE JUST LOOKING FOR THE MOST OF A CATEGORY
                    group_counts = [player[1][np.where(player[0] == group)] for player in val_counts]
                    group_counts = [np.append(player, 0)[0] if len(player) == 0 else player[0] for player in group_counts]
                    best_player = SushiDraft.winningPlayer(group_counts)
                    if best_player is None:
                        continue
                    self.passScoreToken(best_player, group)
            if self.num_round == 3: # This is for the case where we've finished the 3rd round
                self.num_cards_played = 0
                # EXIT GAME
                if self.suppressPrint == False:
                    print("GAME IS OVER")
                    print("Results:")
                    print(self.player_tokens)
                return 0
            # RESET THE GAME NOW THAT POINTS HAVE BEEN HANDED OUT
#            np.random.seed(1)
            self.num_round += 1
            if len(self.one_player_hands) != 0:# Again, this is when we want to initialize our player with a specific hand
                new_deck = deck.copy()
                [new_deck.remove(card) for card in self.one_player_hands[self.num_round - 1]]
                self.hand_cards = np.array_split(np.random.choice(new_deck, 6 * (self.num_players - 1), False), self.num_players - 1)
                self.hand_cards.insert(0, self.one_player_hands[self.num_round - 1])
            else:
                self.hand_cards = np.array_split(np.random.choice(deck, 6 * self.num_players, False), self.num_players)
            self.num_cards_played = 0
            self.played_cards = [np.empty(0, dtype='int') for i in range(self.num_players)]
        else: # This is the portion where you save a card in your hand and pass, etc.
            for i in range(len(self.hand_cards)): # Make sure that we correctly account for wildcards
                if is_wild[i]:
                    play_cards[i] = 2
            # UPDATE HANDS AFTER PLAYING CARD
            self.hand_cards = [np.delete(self.hand_cards[i], np.where(self.hand_cards[i] == play_cards[i])[0][0]) for i in range(len(self.hand_cards))]
            # UPDATE HANDS AFTER SAVING CARD
            self.hand_cards = [np.delete(self.hand_cards[i], np.where(self.hand_cards[i] == save_cards[i])[0][0]) for i in range(len(self.hand_cards))]
            # PASS THE CARDS
            self.hand_cards = [self.hand_cards[i-1] if i > 0 else self.hand_cards[self.num_players - 1] for i in range(len(self.hand_cards))]
            # ADD THE SAVED CARD TO THE HAND
            self.hand_cards = [np.append(self.hand_cards[i], save_cards[i]) for i in range(len(self.hand_cards))]
        return 1
    
    def winningPlayer(group_counts):
        # Take as input the counts for each player of their cards for any of the specific 
        # scoring categories. Then return the winning player (or None if there isn't any)
        unique, counts = np.unique(np.asarray(group_counts), return_counts = True)
        # Get the people with NO ties
        no_ties = unique[np.where(counts == 1)]
        if len(no_ties) == 0:
            return None # No points awarded; no one had a unique diversity
        if len(no_ties) == 1 and no_ties[0] == 0:
            return None # No points awarded; no one had a unique diversity
        # Now get the best player
        best_count = np.sort(no_ties)[-1]
        best_player = np.where(group_counts == best_count)[0]
        return best_player
    def passScoreToken(self, winner, category):
        # You pass a winning player (a single number 0-5) and an associated category 
        # (a single number 2, 4:8), and then give that player a random point from 
        # that category. Then update the score_tokens_avail
        if type(self.score_tokens_avail[category]) == np.int64: # For whatever reason np.random.choice does not work correctly on
            # a 1-element array
            token_won = self.score_tokens_avail[category]
        else:
            token_won = np.random.choice(self.score_tokens_avail[category], 1)[0]
        # Give the winner the token at random
        self.player_tokens[winner] += token_won
        remove_this_token = Series(token_won, [category])
#        remove_this_loc = np.asarray(list(remove_this_token.index)[0] == list(dummy.score_tokens_avail.index)) & \
#            np.asarray(list(remove_this_token.values)[0] == list(dummy.score_tokens_avail.values))
        # Find the tokens in the set of available tokens that match the one that was selected (match the index and value)
        remove_this_loc = np.where(list(remove_this_token.index)[0] == self.score_tokens_avail.index, True, False) & \
            np.where(list(remove_this_token.values)[0] == self.score_tokens_avail.values, True, False)
        # Ensure only first True value used
        remove_this_loc = remove_this_loc & np.cumsum(remove_this_loc) == 1 
        # Remove the score tokens out of circulation
        self.score_tokens_avail = self.score_tokens_avail[np.logical_not(remove_this_loc)]
    
    def __str__(self):
        return 'Sushi Draft game in round ' + str(self.num_round) + ' with ' \
                + str(self.num_players) + ' players and ' + str(self.num_cards_played) + ' cards played.' \
                + ' The hands consist of ' + str(self.hand_cards)
                
# Getting back the possStateActions object:
f=open('/Users/Pan/Google Drive/Data Science/Reinforcement Learning/ReinforcementLearning-SushiDraft/possStateActions.pkl','rb')  
possStateActions = pickle.load(f)
f.close()

possStateActions['Q'] = 0
possStateActions['state'] = possStateActions['state'].apply(str)   
#possStateActions['state'][12000]
'''
NEW FUNCTION 1
'''
def get_others_boolean(played_cards):

    played=currentState(played_cards[0])
    
    others=[]
    for i in range(1,len(played_cards)):
        others.append(currentState(played_cards[i]))
        
    # max other
    other=list(map(max,list(zip(*others))))
    
    other_bool = []
    for x, y in zip(played, other):
        if x>y:
            other_bool.append(-1)
        elif y == x:
            other_bool.append(0)
        else:
            other_bool.append(1)
    
    return other_bool

'''
UPDATED FUNCTION 1
'''
def evaluatePolicy(num_iters, num_trials, qStateActionSpace, numPlayers, method, policySpace = None):
    """
    Many of our functions are performing policy control and, during that process, 
    evaluating the performance of our policy we've been training. Ideally we can 
    just put all of that code into one function that is called by every other function.
    
    It takes how many trials of the game
    """

    # Initialize some counts of to keep track of winners
    win_counts = Series([0] * numPlayers, range(numPlayers))
    # We can use these next two to figure out what percentage of the state-space we actually visited across out trials
    no_value = 0 
    total_values = 0
    # Run the optimal policy across num_trials number of games
    for j in range(num_trials):# Play the game some set number of times
        dummy = SushiDraft(1, numPlayers, score_tokens, deck, 0) # random initialization of the game
        isPlaying = 1
        while(isPlaying):
            total_values += 1
            curr_played_cards=dummy.played_cards
            currState = [currentState(dummy.hand_cards[0]),
                         currentState(curr_played_cards[0]),get_others_boolean(curr_played_cards)]
            # Selecting the possible actions corresponding to this current state
            
            qStateActionSpace['state'] = qStateActionSpace['state'].apply(str)
            possActions = qStateActionSpace[qStateActionSpace['state'] == str(currState)]
  
                
            # Figure out what proportion of states haven't been visited
            
            if possActions.sample(len(possActions))['Q'].max() == 0:
                    no_value += 1
            
            # Now decide which action to take -- follow optimal
            piActionIndex = possActions.sample(len(possActions))['Q'].idxmax()
            
            # Now record what our character is going to do
            play_card, keep_card, is_wildcard = possActions.loc[piActionIndex]['action']
            
            # Figure out what the competition is going to do
            if policySpace is None: # Use the random agent to play
                play_cards, keep_cards, is_wildcards = randomMoves(dummy.hand_cards, dummy.played_cards, range(1, dummy.num_players))
            else: # Use a trained agent to play
                play_cards, keep_cards, is_wildcards = policyMoves(dummy.hand_cards, dummy.played_cards, policySpace, range(1, dummy.num_players))
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
    win_percent['player'] = range(numPlayers)
    win_percent['nTrainIter'] = num_iters
    win_percent['winPercent'] = win_percent['nWins'] / win_percent['nTrials']
    # Append a method for the sake of remembering what we did
    win_percent['method'] = method
    win_percent = win_percent[['method', 'player','nTrainIter', 'nTrials', 'nWins', 'winPercent']]
    return win_percent

'''
UPDATED FUNCTION 2: qLearning
'''

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
                         #sample currState: [[0, 2, 1, 0, 2, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
                         
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

# Run example
qStateActionSpace,_=qLearning(possStateActions)

# RUN this everytime when passing qStateActionSpace as an parameter !!!!
qStateActionSpace['state'] = qStateActionSpace['state'].apply(str) 
qLearning(possStateActions, evalPolicySpace = qStateActionSpace)
qStateActionSpace, win_percents = qLearning(qStateActionSpace.drop(['method'], axis = 1),
                                            measureWinPoints = np.asarray([1]), 
                                            numIterations = np.asarray([1000]))