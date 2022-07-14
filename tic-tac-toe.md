---
title: Tic Tac Toe
layout: default
filename: tic-tac-toe.md
--- 

## Tic Tac Toe

Who doesn't love a good game of tic tac toe? In this project, we will use reinforcement learning, more specifically Q-learning, to teach an agent how to play tic tac toe.

This project will require a good amount of extra code to run the game, so I will provide you that code. To finish the reinforcement learning part of this project, search for the keyword 'TODO', which signifies where you will need to work in the file.

A summary of the tasks:

- Assign the rewards after a game is finished
    - Use an if statement to check whether the result is 1, -1, or any other value
        - A result of 1 means the AI/computer won
        - A result of -1 means the Human won
        - Any other result means the game ended in a tie
    - Reward a win with a value of 1 and a loss with a value of 0
        - For rewarding the AI/computer use self.p1.feedReward(reward)
        - For rewarding the AI/computer use self.p2.feedReward(reward)
- Complete the equation for feeding reward
    - The equation should look like: ``` current_state_value = current_state_value + learning_rate * ((decay_gamma * reward) - current_state_value) ```
        -  The current state value to be updated is ```self.states_value[st]```
        -  Learning rate is ```self.lr```
        -  Decay gamma is ```self.decay_gamma```

Download the code [here](tictactoe.py)

***

This exercise was adapted from [towardsdatascience](https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542)
