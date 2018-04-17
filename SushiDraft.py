#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:20:58 2018

@author: mead
"""
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from itertools import permutations

# SUSHI DRAFT

# Number of players
nPlayers = 5
# The deck of cards
deck = [8, 8, 8, 8, 8, 8, 8, 8,
        7, 7, 7, 7, 7, 7, 7, 
        6, 6, 6, 6, 6, 6,
        5, 5, 5, 5, 5,
        4, 4, 4, 4,
        2, 2]
# Possible score tokens according to diversity(2), grey sushi(4), yellow sushi(5), 
# blue sushi(6), green sushi(7), and red sushi(8)
score_tokens = Series([3, 4, 5, 2, 4, 4, 2, 2, 4, 1, 3, 3, 1, 2, 3, 2, 3, 5], 
                      [8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 2, 2, 2])



##########################
####                 #####
####    CLASS DEF    #####
####                 #####
##########################   
class SushiDraft:
    """
    The SushiDraft class embodies an instance of the game. It initializes a configuration
    of the game based on the parameters passed to it. By default, the behavior is to 
    initialize a random game at the very beginning of play.
    
    The primary way of interacting with the SushiDraft instance from outside of the class
    is with the takeTurn function. takeTurn() is passed cards to play for each player, cards to
    save for each player, and if any of the cards played are being done as a wildcard
    """
    def __init__(self, num_round, num_players, score_tokens_avail, deck, num_cards_played, hand_cards = [], played_cards = [], player_tokens = Series(), one_player_hands = []):  # constructor
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

##########################
####                 #####
####    FUNCTIONS    #####
####                 #####
##########################   
def randomMoves(hands, already_played_cards, random_players = [1, 2, 3, 4]):
    """
    Takes a list of cards in hands and cards already played (usually from the SushiDraft attributes)
    And returns moves under the random policy. random_players can allow you to control whether or not a 
    certain players moves are made as random. NOTE: player 0 is usually the one we're training.
    """
    play_cards, keep_cards, is_wildcards = ([], [], [])
#    for i in range(len(hands)):
    for i in random_players:
        # Use the randomPolicy() 
        play_card, keep_card, is_wildcard = randomPolicy(hands[i], already_played_cards[i])
        play_cards.append(play_card)
        keep_cards.append(keep_card)
        is_wildcards.append(is_wildcard)
    return (play_cards, keep_cards, is_wildcards)

def randomPolicy(hand, already_played_cards):
    """
    fields: hand is an ndarray of cards with possible values of [2, 4:8], with anywhere from 2-6 members
            alread_played_cards is a list of cards that the player has already played in front of them
    outputs: values for the cards to be played, saved, and if it's a wildcard
    """
    is_wildcard = 0
    if len(hand) == 6: #ie: first move -- can't use wildcard (must copy something)
        play_card = np.random.choice(hand[hand != 2])
    else:
        play_card = np.random.choice(hand)
    if play_card == 2: # mark as wildcard and pick the wildcard to be some other card
        play_card = np.random.choice(np.unique(already_played_cards))
        is_wildcard = 1
    if is_wildcard: # make sure not to let yourself save the wildcard after using it
        keep_card = np.random.choice(np.delete(hand, [np.argmax(hand == 2)]))
    else:
        keep_card = np.random.choice(np.delete(hand, [np.argmax(hand == play_card)]))
    return (play_card, keep_card, is_wildcard)

def currentState(hand):
    """
    Interested in specifically defining the state in a way that is useful to us and reduces dimensionality.
    So this is a hand --> state conversion. We need this for going from hand attributes of SushiDraft 
    to the way we think of states
    """
    cardDict = {2 : 0, 4 : 1, 5 : 2, 6 : 3, 7 : 4, 8 : 5} # Matches the card to the index we use
    state = np.zeros(len(cardDict), dtype = 'int').tolist() # Initialize an all-zeroes list of the proper dims
    for card in hand: # now iterate over values of the hand and increase correct indices counts
        state[cardDict[card]] += 1
    return state

def currentHand(state):
    """
    Very similar vein as before. Now take a state as input and output one of the many
    possible hands.
    """
    cardDict = {0 : 2, 1 : 4, 2 : 5, 3 : 6, 4 : 7, 5 : 8} # Matches the card to the index we use
    hand = []
    for i, card in enumerate(state): # Now go through each value in the state
        for number in range(card): # and append the card value the appropriate number of times
            hand.append(cardDict[i])
    return hand
    
def possibleActions(state, already_played_cards):
    """
    We are interested in the possible moves because this is what we will add to
    the policy control portion (the Q state-action pairs). What's returned is a 
    list of lists. Each list is a move that can be taken. The elements of each 
    list are [card played, card saved, is_wildcard] which pertain to all the possible 
    moves associated with being in the given state
    """
    cardDict = {0 : 2, 1 : 4, 2 : 5, 3 : 6, 4 : 7, 5 : 8} # Matches the card to the index we use
    hand = currentHand(state)
    moves = np.unique([list(combo) for combo in permutations(hand, 2)], axis = 0).tolist()
    [move.append(0) for move in moves] # This gets us an exhaustive list of moves. Note that some are invalid though
    for i, move in reversed(list(enumerate(moves))): # Need to get rid of unallowed moves. Eg: Playing a wildcard first turn
        if move[0] == 2: # So playing a wildcard
#            moves = moves[np.isin(moves, move)]
            del moves[i] # Get rid of the case where we play a wildcard
#            print(moves)
            for played in np.where(np.asarray(already_played_cards) != 0)[0]: # And replace it with appropriate uses of the wildcard
                if played.size != 0: # Check for the case where no cards have been played
                    moves.append([cardDict[played], move[1], 1])
    # Get rid of possible duplicated
    moves = np.unique(moves, axis = 0).tolist()
    return moves

def getWinner(outcome):
    """
    Take as input the outcome of a game (passed in the same form as the 
    SushiDraft player_tokens attribute). Returns a value of 1 for each of the 
    winning players (including ties) and a 0 for any losing players
    """
    maxVal = outcome.max()
    outcome[outcome < maxVal] = 0
    outcome[outcome == maxVal] = 1
    return outcome


################################
####                       #####
####    WORKING EXAMPLE    #####
####                       #####
################################  
# Example code for running a full game
dummy = SushiDraft(1, 5, score_tokens, deck, 0) # random initialization of the game
isPlaying = 1
while (isPlaying):
#    print(dummy)
    play_cards, keep_cards, is_wildcards = randomMoves(dummy.hand_cards, dummy.played_cards, [0, 1, 2, 3, 4])
    isPlaying = dummy.takeTurn(play_cards, keep_cards, is_wildcards)




################################
####                       #####
####    STATE SPACE DEF    #####
####                       #####
################################  
# Okay. So to make this work, we need to create state-space representations
# List of cards that could be in a hand
handStates = []
for a in [0, 1, 2]: # Wildcards
    for b in [0, 1, 2, 3, 4]: # 4's
        for c in [0, 1, 2, 3, 4, 5]: # 5's
            for d in [0, 1, 2, 3, 4, 5, 6]: # 6's
                for e in [0, 1, 2, 3, 4, 5, 6]: # 7's
                    for f in [0, 1, 2, 3, 4, 5, 6]: # 8's
                        if 2 <= a + b + c + d + e + f & a + b + c + d + e + f <= 6:
                            handStates.append([a, b, c, d, e, f])
len(handStates)

# Need to add in all of the possible states that I may have alreadyPlayed <===Important
playedStates = []
for a in [0]: # Wildcards
    for b in [0, 1, 2, 3, 4, 5]: # 4's
        for c in [0, 1, 2, 3, 4, 5]: # 5's
            for d in [0, 1, 2, 3, 4, 5]: # 6's
                for e in [0, 1, 2, 3, 4, 5]: # 7's
                    for f in [0, 1, 2, 3, 4, 5]: # 8's
                        if 0 <= a + b + c + d + e + f & a + b + c + d + e + f <= 5:
                            playedStates.append([a, b, c, d, e, f])
len(playedStates)

# This builds the full state-space for the scenario in which we only consider 
# the player's hand and what they've already played
possStates = []
for hand in handStates:
    for played in playedStates:
        if sum(hand) + sum(played) == 6:
            possStates.append([hand, played])
len(possStates)


# THIS PART IS HUGE. I'm constructing a comprehensive state-action space to hold 
# our estimates of Q-value
possStateActions = DataFrame()
for i in range(len(possStates)): # Run through each of the states we've identified
    print(i)
    possStateAction = DataFrame()
    # And now identify every single possible action for that state
    possStateAction['action'] = possibleActions(possStates[i][0], possStates[i][1])
    # Note that in order to get this to work, I needed to make the state a string
    possStateAction['state'] = str(possStates[i])
    possStateActions = possStateActions.append(possStateAction)
possStateActions = possStateActions.reset_index()
possStateActions = possStateActions.drop('index', axis = 1)
possStateActions['Q'] = 0
possStateActions = possStateActions[['state','action','Q']]




##################################
####                         #####
####    Q-Learning Control   #####
####                         #####
##################################
def qLearning(qStateActionSpace, epsilon = .9, alpha = .5, gamma = 1, 
              measureWinPoints = np.asarray([10, 20]), numIterations = np.asarray([20, 30]),
              numPlayers = 5):
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
    measureWinPoints = np.asarray(measureWinPoints)
    numIterations = np.asarray(numIterations)
    win_percents = DataFrame() # Track the win percentages across players (draws count as wins)
    for i in range(1, max(measureWinPoints) + 1): # Perform the algorithm as many times as we want
        totalReward = 0
        dummy = SushiDraft(1, numPlayers, score_tokens, deck, 0) # random initialization of the game
        isPlaying = 1
        while(isPlaying): # Run through one full game, peforming control as we go
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
                dummy = SushiDraft(1, numPlayers, score_tokens, deck, 0) # random initialization of the game
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
    win_percents['method'] = 'q-learning'
    win_percents = win_percents[['method', 'player','nTrainIter', 'nTrials', 'nWins', 'winPercent']]
    qStateActionSpace['method'] = 'q-learning'
    qStateActionSpace = qStateActionSpace[['method', 'state', 'action', 'Q']]
    # Return the q-state action values and our optimal policy win rates
    return (qStateActionSpace, win_percents)

# Run example
#qStateActionSpace, win_percents = qLearning(possStateActions)
#
#qStateActionSpace, win_percents = qLearning(qStateActionSpace.drop(['method'], axis = 1),
#                                            measureWinPoints = np.asarray([1]), 
#                                            numIterations = np.asarray([1000]))



############################################
####                                   #####
####    Monte Carlo Exploring Starts   #####
####                                   #####
############################################
def monteCarloES(qStateActionSpace, epsilon = .9, alpha = 0, gamma = 1, 
                 measureWinPoints = np.asarray([10, 20]), numIterations = np.asarray([20, 30]), 
                 possibleInitialStates = [state for state in handStates if sum(state) == 6],
                 numPlayers = 5):
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
    measureWinPoints = np.asarray(measureWinPoints)
    numIterations = np.asarray(numIterations)
    # Pick the following hands for our player to start out with (with equal probabilities)
    possibleInitialHands = [currentHand(state) for state in possibleInitialStates]
    
    qStateActionSpace['N_sa'] = 0
    
    win_percents = DataFrame() # Track the win percentages across players (draws count as wins)

    for i in range(1, max(measureWinPoints) + 1): # Perform the algorithm as many times as we want
        totalReward = 0
        
        initialIndices = np.random.choice(range(len(possibleInitialHands)), 3).tolist()
        initialHands = np.asarray(possibleInitialHands)[initialIndices]
        initialStates = np.asarray(possibleInitialStates)[initialIndices]
        
        # Pick a random action associated with being in that starting state
        initialActionIndices = [qStateActionSpace[qStateActionSpace['state'].str.slice(1,19) == str(state.tolist())].sample(1).index.tolist()[0] for state in initialStates]
        qStateActionSpace.iloc[initialActionIndices]
        
        dummy = SushiDraft(1, numPlayers, score_tokens, deck, 0, one_player_hands = initialHands) # random initialization of the game
        isPlaying = 1
        
        # Keep track of the episodes as they are played
        episodeTracker = DataFrame()
        
        while(isPlaying): # Run through one full game, peforming control as we go
            currState = [currentState(dummy.hand_cards[0]),
                         currentState(dummy.played_cards[0])]
            # Selecting the possible actions corresponding to this current state
            possActions = qStateActionSpace[qStateActionSpace['state'] == str(currState)]
            # Epsilon-greedy implementation
            greedy_prob = 1 - epsilon
            # Now decide which action to take
            if dummy.num_cards_played == 0: # In the first round take the action we randomly initialized
                print("Taking randomly initialized action")
                actionIndex = initialActionIndices[dummy.num_round - 1]
                # Now record what our character is going to do
                play_card, keep_card, is_wildcard = qStateActionSpace.loc[actionIndex,'action']
                qStateActionSpace.loc[actionIndex, 'N_sa'] += 1
            else: # For all other turns consider the 
                if np.random.random() < greedy_prob:
                    # Take the greedy action
                    actionIndex = possActions.sample(len(possActions))['Q'].idxmax()
                else:
                    # Take a random action
                    actionIndex = possActions.sample(1)['Q'].idxmax()
                # Now record what our character is going to do
                play_card, keep_card, is_wildcard = possActions.loc[actionIndex]['action']
                qStateActionSpace.loc[actionIndex, 'N_sa'] += 1

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
            
            # ADDED
            episodeTracker = episodeTracker.append(DataFrame({'stateActionIndex' : actionIndex, 'reward' : immedReward}, index = [0]))
        episodeTracker = episodeTracker.reset_index()

        # Assign memory to tracking rewards
        cumRewards = pd.Series(0, index = episodeTracker['stateActionIndex'])
        # Need to perform the cumulative sum
        # We're using a slightly modified gamma discount that is 0 outside a specific round
        for j in range(15):
            rewardsOfInterest = np.asarray(episodeTracker[j : 5 * np.ceil((j+1)/5).astype('int')]['reward']) # Fancy ceiling to do what I want
            gammaVals = np.geomspace(1, 
                                     gamma ** (len(rewardsOfInterest) - 1), 
                                     len(rewardsOfInterest))
            cumRewards.iloc[j] = np.sum(rewardsOfInterest * gammaVals)
        
        # Now do the updates
        if alpha == 0:
            qStateActionSpace.loc[episodeTracker['stateActionIndex'], 'Q'] += (cumRewards - qStateActionSpace.iloc[episodeTracker['stateActionIndex']]['Q']) / qStateActionSpace.iloc[episodeTracker['stateActionIndex']]['N_sa']
        else:
            qStateActionSpace.loc[episodeTracker['stateActionIndex'], 'Q'] += alpha * (cumRewards - qStateActionSpace.iloc[episodeTracker['stateActionIndex']]['Q'])

        
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
                dummy = SushiDraft(1, numPlayers, score_tokens, deck, 0) # random initialization of the game
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
    win_percents['method'] = 'monte-carlo-exploring-starts'
    win_percents = win_percents[['method', 'player','nTrainIter', 'nTrials', 'nWins', 'winPercent']]
    qStateActionSpace['method'] = 'monte-carlo-exploring-starts'
    qStateActionSpace = qStateActionSpace[['method', 'state', 'action', 'Q']]
    # Return the q-state action values and our optimal policy win rates
    return (qStateActionSpace, win_percents)


qStateActionSpace, win_percents = monteCarloES(possStateActions,
                                            measureWinPoints = np.asarray([1000]), 
                                            numIterations = np.asarray([1000]))




############################################
####                                   #####
####    Monte Carlo Exploring Starts   #####
####                                   #####
############################################
def sarsa_lambda(qStateActionSpace, epsilon=.9, alpha = .5, gamma = 1, lambda_ = .1, 
                 measureWinPoints = np.asarray([10, 20]), numIterations = np.asarray([20, 30]), 
                 numPlayers = 5, score_tokens):
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
#    try:
#        qStateActionSpace = qStateActionSpace.drop('method', 1)
#    except:
#        print("No method column drop")
#    measureWinPoints = np.asarray(measureWinPoints)
#    numIterations = np.asarray(numIterations)
    
    win_percents = DataFrame() # Track the win percentages across players (draws count as wins)
    for i in range(1, max(measureWinPoints) + 1): # for every episode
        print(i)
        totalReward = 0
        dummy = SushiDraft(1, numPlayers, score_tokens, deck, 0) # random initialization of the game
        isPlaying = 1
        
        ########################
        ###  ADRIAN COMMENT  ###
        ########################
        # I was thinking that we could reset the eligibility traces back to 0 here, 
        # at the beginning of an episode, but I was also thinking that we could do
        # it at the beginning of each of the 3 rounds...
        qStateActionSpace['E'] = 0

        
        while(isPlaying): # while not terminate, play a round
#            print(isPlaying)
            # Need to take care of initial action selection
            if dummy.num_cards_played == 0:
                currState = [currentState(dummy.hand_cards[0]),currentState(dummy.played_cards[0])]
                # Selecting the possible actions corresponding to this current state
                possActions = qStateActionSpace[qStateActionSpace['state'] == str(currState)]
                # Epsilon-greedy implementation
                greedy_prob = 1 - epsilon

                # Reset the eligibility trace
                qStateActionSpace['E'] = 0
                # Now decide which action to take -- only done this way for the first move each round
                if np.random.random() < greedy_prob:
                    # Take the greedy action
                    muActionIndex = possActions.sample(len(possActions))['Q'].idxmax()
                else:
                    # Take a random action
                    muActionIndex = possActions.sample(1)['Q'].idxmax()
            # Otherwise, we just use the actions selected in the following steps
            
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
                # Need to do epsilon-greedy again
                if np.random.random() < greedy_prob:
                    # Take the greedy action
                    piNextActionIndex = possNextActions.sample(len(possNextActions))['Q'].idxmax()
                else:
                    # Take a random action
                    piNextActionIndex = possNextActions.sample(1)['Q'].idxmax()
                
                # PERFORM THE Q-update
                delta = immedReward + gamma * qStateActionSpace.loc[piNextActionIndex, 'Q'] - qStateActionSpace.loc[muActionIndex, 'Q']
                qStateActionSpace.loc[muActionIndex, 'E'] += 1
                
                ########################
                ###  ADRIAN COMMENT  ###
                ########################
                qStateActionSpace['Q'] += alpha * delta * qStateActionSpace['E']
                qStateActionSpace['E'] += gamma * lambda_ * qStateActionSpace['E']
                
                # Now set the next state/action to be the current state/action
                currState = nextState.copy()
                muActionIndex = piNextActionIndex.copy()
                possActions = possNextActions.copy()
#                update_these = np.where(E['Q'] != 0)[0]
#                
#                for s in update_these:  
#                    print(s)
#                    qStateActionSpace.loc[s, 'Q'] += alpha * delta * E.loc[s, 'Q']
#                    E.loc[s,'Q']=gamma*lambda_*E.loc[s,'Q']
            else:
    #            print("End of the round")
    #            print("Immediate Reward: " + str(immedReward))
                # This is the case where the round was finished
                # Think about if we want to keep it this way. We're basically saying the terminal state has value 0, which is probably reasonable
                delta = immedReward + gamma * 0 - qStateActionSpace.loc[muActionIndex, 'Q']
                qStateActionSpace.loc[muActionIndex, 'E'] += 1
                # Go through and update the Q and E values
                qStateActionSpace['Q'] += alpha * delta * qStateActionSpace['E']
                qStateActionSpace['E'] += gamma * lambda_ * qStateActionSpace['E']

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
                dummy = SushiDraft(1, numPlayers, score_tokens, deck, 0) # random initialization of the game
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
    qStateActionSpace['method'] = 'sarsa_lambda'
    qStateActionSpace = qStateActionSpace[['method', 'state', 'action', 'Q']]
    # Return the q-state action values and our optimal policy win rates
    return (qStateActionSpace, win_percents)

qStateActionSpace, win_percents = sarsa_lambda(qStateActionSpace, measureWinPoints=[1000], numIterations=[1000])
# Train over 2000 games and simulate 1000 games
#         method  player  nTrainIter  nTrials  nWins  winPercent
#0  sarsa_lambda       0        1000     1000    336       0.336
#1  sarsa_lambda       1        1000     1000    238       0.238
#2  sarsa_lambda       2        1000     1000    172       0.172
#3  sarsa_lambda       3        1000     1000    176       0.176
#4  sarsa_lambda       4        1000     1000    186       0.186




## Only considering what's in the hand
#possStateActions = DataFrame()
#for i in range(len(handStates)): # Run through each of the states we've identified
#    print(i)
#    possStateAction = DataFrame()
#    # And now identify every single possible action for that state
#    if handStates[i] == [2, 0, 0, 0, 0, 0] or handStates[i] == [1, 0, 0, 0, 0, 0]:
#        continue
#    possStateAction['action'] = possibleActions(handStates[i], [0, 0, 0, 0, 0, 0])
#    # Note that in order to get this to work, I needed to make the state a string
#    possStateAction['state'] = str(handStates[i])
#    possStateActions = possStateActions.append(possStateAction)
#possStateActions = possStateActions.reset_index()
#possStateActions = possStateActions.drop('index', axis = 1)
#possStateActions['Q'] = 0
#possStateActions = possStateActions[['state','action','Q']]
#
#
#qHand, handWinPercents = qLearning(possStateActions,
#                                   measureWinPoints = np.asarray([100, 500, 1000, 2000]),
#                                   numIterations = np.asarray([250, 250, 250, 1000]))