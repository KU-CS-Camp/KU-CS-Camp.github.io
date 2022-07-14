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

Replace your play() function with the following code:
```
    def play(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
```

***

This exercise was adapted from [towardsdatascience](https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542)
