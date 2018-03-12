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

nPlayers = 5
deck = [8, 8, 8, 8, 8, 8, 8, 8,
        7, 7, 7, 7, 7, 7, 7, 
        6, 6, 6, 6, 6, 6,
        5, 5, 5, 5, 5,
        4, 4, 4, 4,
        2, 2]
score_tokens = Series([3, 4, 5, 2, 4, 4, 2, 2, 4, 1, 3, 3, 1, 2, 3, 2, 3, 5], 
                      [8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 2, 2, 2])
        

class SushiDraft:
    # fields: num_round, num_players, score_tokens (a Series), deck (a list), num_cards_played (from 0 to 4 when initializing), hand_cards (a list of ndarrays), played_cards (a list of ndarrays), and player_tokens (a Series)
    def __init__(self, num_round, num_players, score_tokens_avail, deck, num_cards_played, hand_cards = [], played_cards = [], player_tokens = Series()):  # constructor
        self.num_round = num_round
        self.num_players = num_players
        self.score_tokens_avail = score_tokens_avail
        self.deck = deck
        self.num_cards_played = num_cards_played
        if len(hand_cards) == 0:
            # This is basically the case where you shuffle the deck
#            np.random.seed(1)
            self.hand_cards = np.array_split(np.random.choice(deck, 6 * num_players, False), num_players)
        else:
            self.hand_cards = hand_cards
        if len(played_cards) == 0:
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
            if self.num_round == 3:
                # EXIT GAME I SUPPOSE
                print("GAME IS OVER")
                print("Results:")
                print(self.player_tokens)
                return 0
            # RESET THE GAME NOW THAT POINTS HAVE BEEN HANDED OUT
#            np.random.seed(1)
            self.num_round += 1
            self.hand_cards = np.array_split(np.random.choice(deck, 6 * self.num_players, False), self.num_players)
            self.num_cards_played = 0
            self.played_cards = [np.empty(0, dtype='int') for i in range(num_players)]
        else: # This is the portion where you save a card in your hand and pass, etc.
            for i in range(len(self.hand_cards)): # Make sure that we correctly account for wildcards
                if is_wild[i]:
                    play_cards[i] = 2
            # UPDATE HANDS AFTER PLAYING CARD
            self.hand_cards = [np.delete(self.hand_cards[i], np.where(self.hand_cards[i] == play_cards[i])[0][0]) for i in range(len(self.hand_cards))]
            # UPDATE HANDS AFTER SAVING CARD
            self.hand_cards = [np.delete(self.hand_cards[i], np.where(self.hand_cards[i] == save_cards[i])[0][0]) for i in range(len(self.hand_cards))]
            # PASS THE CARDS CLOCKWISE
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
        self.player_tokens[winner] += token_won
        remove_this_token = Series(token_won, [category])
        remove_this_loc = np.asarray(list(remove_this_token.index)[0] == list(self.score_tokens_avail.index)) & \
            np.asarray(list(remove_this_token.values)[0] == list(self.score_tokens_avail.values))
        remove_this_loc = remove_this_loc & np.cumsum(remove_this_loc) == 1 # Ensure only first True value used
        self.score_tokens_avail = self.score_tokens_avail[np.logical_not(remove_this_loc)]
    
    def __str__(self):
        return 'Sushi Draft game in round ' + str(self.num_round) + ' with ' \
                + str(self.num_players) + ' players and ' + str(self.num_cards_played) + ' cards played.' \
                + ' The hands consist of ' + str(self.hand_cards)



def playerMoves(hands, already_played_cards):
    """
    Takes a list of cards in hands and cards already played (usually from the SushiDraft attributes)
    And returns moves under the random play policy
    """
    play_cards, keep_cards, is_wildcards = ([], [], [])
    for i in range(len(hands)):
        play_card, keep_card, is_wildcard = randomPolicy(hands[i], already_played_cards[i])
        play_cards.append(play_card)
        keep_cards.append(keep_card)
        is_wildcards.append(is_wildcard)
    return (play_cards, keep_cards, is_wildcards)

def currentState(hand):
    """
    Interested in specifically defining the state in a way that is useful to us and reduces dimensionality
    """
    cardDict = {2 : 0, 4 : 1, 5 : 2, 6 : 3, 7 : 4, 8 : 5} # Matches the card to the index we use
    state = np.zeros(len(cardDict), dtype = 'int').tolist()
    for card in hand:
        state[cardDict[card]] += 1
    
def possibleActions(state, already_played_cards):
    """
    We are interested in the possible moves because this is what we will add to the MonteCarlo (exhaustive search sort of thing)
    """
    cardDict = {0 : 2, 1 : 4, 2 : 5, 3 : 6, 4 : 7, 5 : 8} # Matches the card to the index we use
    hand = []
    for i, card in enumerate(state):
        for number in range(card):
            hand.append(cardDict[i])
    moves = np.unique([list(combo) for combo in permutations(hand, 2)], axis = 0).tolist()
    [move.append(0) for move in moves]
    for i, move in reversed(list(enumerate(moves))):
        if move[0] == 2: # So playing a wildcard
#            moves = moves[np.isin(moves, move)]
            del moves[i]
#            print(moves)
            for played in already_played_cards:
                moves.append([played, move[1], 1])
    moves = np.unique(moves, axis = 0)
    return moves

    
def randomPolicy(hand, already_played_cards):
    # hand is an ndarray of cards (2, 4:8), with anywhere from 2-6 members
    # alread_played_cards is a list of cards that the player has already played in front of them
    # outputs something that could be combined with other players to be played intelligibly with the game
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

dummy = SushiDraft(1, 5, score_tokens, deck, 0) # random initialization of the game
isPlaying = 1
while (isPlaying):
#    print(dummy)
    play_cards, keep_cards, is_wildcards = playerMoves(dummy.hand_cards, dummy.played_cards)
    isPlaying = dummy.takeTurn(play_cards, keep_cards, is_wildcards)



# Need to add in all of the possible states that I may have alreadyPlayed <===Important
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

playedStates = []
for a in [0, 1, 2]: # Wildcards
    for b in [0, 1, 2, 3, 4]: # 4's
        for c in [0, 1, 2, 3, 4, 5]: # 5's
            for d in [0, 1, 2, 3, 4, 5]: # 6's
                for e in [0, 1, 2, 3, 4, 5]: # 7's
                    for f in [0, 1, 2, 3, 4, 5]: # 8's
                        if 0 <= a + b + c + d + e + f & a + b + c + d + e + f <= 5:
                            playedStates.append([a, b, c, d, e, f])
len(playedStates)

fullStates = []
for hand in handStates:
    for played in playedStates:
        if sum(hand) + sum(played) == 6:
            fullStates.append([hand, played])
len(fullStates)            
            
a = []
b = []    
for val in fullStates:
    a.append(val[0])
    b.append(val[1])
            
for state in allStates:
    possibleActions(state)
    break

policy = np.zeros((1, len(actionDict)))
policy = DataFrame(policy, columns = list(actionDict.keys()))
policy.iloc[:,:] = 1 / len(actionDict)
policy = policy.cumsum(axis = 1)
policy = policy.set_index([['[0, 0]']])









#
## Playing through a single round
#dummy = SushiDraft(1, 5, score_tokens,deck, 0)
#print(dummy)
#dummy.takeTurn([8, 7, 8, 8, 7], [4, 4, 4, 4, 6])
#print(dummy)
#dummy.takeTurn([8, 6, 7, 8, 8], [2, 4, 4, 4, 6], [1, 0, 0, 0, 0])
#print(dummy)
#dummy.takeTurn([8, 7, 6, 6, 8], [6, 4, 4, 4, 5])
#print(dummy)
#dummy.takeTurn([6, 7, 7, 5, 5], [6, 4, 4, 4, 5], [0, 1, 0, 0, 0])
#print(dummy)
#dummy.played_cards
#dummy.takeTurn([6, 6, 7, 4, 5]) # Last Turn
#print(dummy)
#print(dummy.score_tokens_avail)
#print(dummy.player_tokens)
#
#dummy.takeTurn([8, 7, 8, 8, 7], [4, 4, 4, 4, 6])
#print(dummy)
#dummy.takeTurn([8, 6, 7, 8, 8], [2, 4, 4, 4, 6], [1, 0, 0, 0, 0])
#print(dummy)
#dummy.takeTurn([8, 7, 6, 6, 8], [6, 4, 4, 4, 5])
#print(dummy)
#dummy.takeTurn([6, 7, 7, 5, 5], [6, 4, 4, 4, 5], [0, 1, 0, 0, 0])
#print(dummy)
#dummy.played_cards
#dummy.takeTurn([6, 6, 7, 4, 5]) # Last Turn
#print(dummy)
#print(dummy.score_tokens_avail)
#print(dummy.player_tokens)
#
#dummy.takeTurn([8, 7, 8, 8, 7], [4, 4, 4, 4, 6])
#print(dummy)
#dummy.takeTurn([8, 6, 7, 8, 8], [2, 4, 4, 4, 6], [1, 0, 0, 0, 0])
#print(dummy)
#dummy.takeTurn([8, 7, 6, 6, 8], [6, 4, 4, 4, 5])
#print(dummy)
#dummy.takeTurn([6, 7, 7, 5, 5], [6, 4, 4, 4, 5], [0, 1, 0, 0, 0])
#print(dummy)
#dummy.played_cards
#dummy.takeTurn([6, 6, 7, 4, 5]) # Last Turn
#print(dummy)
#print(dummy.score_tokens_avail)
#print(dummy.player_tokens)
#
#dummy = SushiDraft(0, 5, score_tokens,deck, 4, [np.asarray([4, 4]), np.asarray([5, 5]), np.asarray([6, 6]), np.asarray([7, 7]), np.asarray([8, 8])], 
#                                                [np.asarray([4, 4, 4, 4]), np.asarray([5, 5, 5, 5]), np.asarray([6, 6, 6, 6]), np.asarray([7, 7, 7, 7]), np.asarray([8, 8, 8, 8])])   
#
## Cards currently played
#num_players = 5
#a = [np.array([4]),
# np.array([7]),
# np.array([5]),
# np.array([8]),
# np.array([7])]
##a = [np.array([8, 8, 8, 6]),
## np.array([7, 6, 7, 7]),
## np.array([8, 7, 6, 7]),
## np.array([8, 8, 6, 5]),
## np.array([7, 8, 8, 5])]
#b = play_cards
##b = [6, 6, 7, 4, 5]# Next card played
#c = dummy.player_tokens
##c = Series([0 for i in range(5)], 
##       [i for i in range(5)]) # player_tokens
#a = [np.append(a[i], b[i]) for i in range(len(b))]
## Cards in your hand
#d = [np.asarray([5, 8, 4, 7, 6]),
# np.asarray([8, 8, 7, 2, 7]),
# np.asarray([5, 6, 7, 6, 8]),
# np.asarray([6, 6, 4, 6, 8]),
# np.asarray([7, 8, 2, 5, 4])]
##d = [np.asarray([4, 4, 4]), np.asarray([5, 5, 5]), np.asarray([6, 6, 6]), np.asarray([7, 7, 7]), np.asarray([8, 8, 8])]
##b = [4, 5, 6, 7, 8]# Next card played
#e = keep_cards # card_saved
##e = [4, 5, 6, 7, 8] # card_saved
#d = [np.delete(d[i], np.where(d[i] == b[i])[0][0]) for i in range(len(d))] # UPDATE HANDS AFTER PLAYING CARD
#d = [np.delete(d[i], np.where(d[i] == e[i])[0][0]) for i in range(len(d))] # UPDATE HANDS AFTER SAVING CARD
#d = [d[i-1] if i > 0 else d[num_players - 1] for i in range(len(d))] # PASS THE CARDS CLOCKWISE
#d = [np.append(d[i], e[i]) for i in range(len(d))] # ADD THE SAVED CARD TO THE HAND
#
#
#
#val_counts = [np.unique(player, return_counts=True) for player in a]
#for group in np.unique(list(score_tokens.index)): # handle each category scoring one at a time
#    print(group)
#    if group == 2:
#        # THIS IS FOR THE PLAYER WITH THE MOST DIVERSITY
#        group_counts = [len(player[0]) for player in val_counts]
#        unique, counts = np.unique(np.asarray(group_counts), return_counts = True)
#        # Get the people with NO ties
#        no_ties = unique[np.where(counts == 1)]
#        if len(no_ties) == 0:
#            continue # No points awarded; no one had a unique diversity
#        best_divers = np.sort(no_ties)[-1]
#        best_player = np.where(group_counts == best_divers)[0]
#        # GIVE A RANDOM REMAINING TILE TO THE PLAYER
##        passScoreTokens(best_player, group)
#
#        token_won = np.random.choice(score_tokens[group], 1)[0]         
#        c[best_player] += token_won
#        remove_this_token = Series(token_won, [group])
#        remove_this_loc = np.asarray(list(remove_this_token.index)[0] == list(score_tokens.index)) & \
#            np.asarray(list(remove_this_token.values)[0] == list(score_tokens.values))
#        score_tokens = score_tokens[np.logical_not(remove_this_loc)]
#    else:
#        # THIS IS FOR ALL OTHER CASES; WE'RE JUST LOOKING FOR THE MOST NOW
#        group_counts = [player[1][np.where(player[0] == group)] for player in val_counts]
#        group_counts = [np.append(player, 0)[0] if len(player) == 0 else player[0] for player in group_counts]
#        unique, counts = np.unique(np.asarray(group_counts), return_counts = True)
#        # Get the people with NO ties
#        no_ties = unique[np.where(counts == 1)]
#        if len(no_ties) == 0:
#            continue # No points awarded; no one had a unique diversity
#        if len(no_ties) == 1 and no_ties[0] == 0:
#            continue # No points awarded; no one had a unique diversity
#        best_divers = np.sort(no_ties)[-1]
#        best_player = np.where(group_counts == best_divers)[0]
#        # GIVE A RANDOM REMAINING TILE TO THE PLAYER
##        passScoreTokens(best_player, group)
#        if type(score_tokens[group]) == np.int64: # For whatever reason np.random.choice does not work correctly on
#                                          # a 1-element array
#            token_won = score_tokens[group]
#        else:
#            token_won = np.random.choice(score_tokens[group], 1)[0]         
#        c[best_player] += token_won
#        remove_this_token = Series(token_won, [group])
#        remove_this_loc = np.asarray(list(remove_this_token.index)[0] == list(score_tokens.index)) & \
#            np.asarray(list(remove_this_token.values)[0] == list(score_tokens.values))
#        remove_this_loc = remove_this_loc & np.cumsum(remove_this_loc) == 1 # Ensure only first True value used
#        score_tokens = score_tokens[np.logical_not(remove_this_loc)]