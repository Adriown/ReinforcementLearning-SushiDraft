#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:33:09 2018

@author: kia
"""

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

'''
Note: We changed the max to 4
because hand+played = 6, since hand >=2,
then played <=4
'''
playedStates = []
for a in [0]: # Wildcards
    for b in [0, 1, 2, 3, 4,]: # 4's
        for c in [0, 1, 2, 3, 4]: # 5's
            for d in [0, 1, 2, 3, 4]: # 6's
                for e in [0, 1, 2, 3, 4]: # 7's
                    for f in [0, 1, 2, 3, 4]: # 8's
                        if 0 <= a + b + c + d + e + f & a + b + c + d + e + f <= 4:
                            playedStates.append([a, b, c, d, e, f])
len(playedStates)

# Maximum of cards that others played
otherStates = []
for a in [0]: # Wildcards
    for b in [0, 1, 2, 3, 4]: # 4's
        for c in [0, 1, 2, 3, 4]: # 5's
            for d in [0, 1, 2, 3, 4]: # 6's
                for e in [0, 1, 2, 3, 4]: # 7's
                    for f in [0, 1, 2, 3, 4]: # 8's
                        otherStates.append([a, b, c, d, e, f])
len(otherStates)


deck_max = [2,4,5,6,7,8] #The maximum number of each cards , 2 for wildcards, 8 for #8s, etc.


# This builds the full state-space for the scenario in which we only consider 
# the player's hand and what they've already played
possStates = []
for hand in handStates:
    for played in playedStates:
        #The maximum number of each cards , 2 for wildcards, 8 for #8s, etc.
        deck_max = [2,4,5,6,7,8]
        '''
        Here deck_max is the maximum number of cards that OTHER players can hold,
        which means deck_max = the original deck_max subtracts cards on hand and cards the agent
        has played
        '''
        deck_max = [x - y for x, y in zip(deck_max, hand)] #possible cards - cards on my hand
        deck_max = [x - y for x, y in zip(deck_max, played)] #possible cards - cards on my hand - cards I played
        if sum(hand) + sum(played) == 6:
            for other in otherStates:
                # The cards others can play must be <= deck_max
                if all([x >= y for x, y in zip(deck_max, other)]):
                    
                    # sum(played) is the turn of game we are currently in
                    # The maximum number of cards out there must be <= the number of turns
                    # e.g. 1st turn, there can not be [0,2,0,0,0,0]
                    if max(other) <= sum(played):
                    # sum(other) >= sum(played) because the sum of maximum cards others have played must be
                    # larger or equal to the number of rounds.
                    # eg: turn 1, at least one element in other should be greater or equal to 1, because every
                        #player has played 1 card. it is not posibile to have [0,0,0,0,0,0].
                    # sum(other) <= 4*sum(played) because the number of maximum cards others have played
                    # must be smaller or equal to the total number of cards they can play, which is 4 people *
                    # number of turns
                        if sum(other) >= sum(played) & sum(other) <= 4* sum(played):
                            # the following code is to prevent events like following:
                                # at turn 4:  other=[0,2,2,0,0,0]
                                # the total number of cards others *should* have played =4*4=16
                                # but the maximum number of cards for [0,2,2,0,0,0] is 4*#4 and 5*#5 = 9
                            total_num_cards=0
                            for i in range(len(other)): 
                                #calculate totalx number of cards for the cards have been played by others.
                                if other[i] !=0:
                                    total_num_cards+=deck_max[i]
                            # total number of cards + number of available wild cards >= 
                            #   4 other players *number of rounds 
                            if total_num_cards >= 4* sum(played) -deck_max[0]:
                                possStates.append([hand,played,other])
                                #print([hand,played,other])
                                
                            
                
                
len(possStates)


print(possStates[1246870:-1])

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


                        











