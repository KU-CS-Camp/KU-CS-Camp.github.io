---
title: More Python
layout: default
filename: MorePython.md
--- 

## More Python Exercises

### Resources
- [Terminal Commands](https://www.guru99.com/linux-commands-cheat-sheet.html)
- [Python Reference](https://www.w3schools.com/python/)
- [NumPy Reference](https://www.w3schools.com/python/numpy/numpy_intro.asp)


***
### Guess a Number Game
Create a program that will make the user guess a random number between 1 and 100. The game won't end until the user guesses the correct number, but along the way you will help the user.

Until the user guesses the secret number:
- Obtain a guess
- If it's higher than the number, tell the user
- If it's lower than the number, tell the user
- If they guess the number, the game stops and they are prompted to either play again or exit

After they win...
- Congratulate them
- Tell them how many guesses it took

***
### Wordle
Create a game similar to Wordle.

- Create an array that stores a bunch of words that will be used as the secret word. You can use only 5 letter words (or make different levels that use 6 or 7 letters).
- Prompt the user to enter a word
    - If they enter a word that is too long, tell them
    - If they enter a word that is too short, tell them
    - If their word is the correct length:
        - User guesses incorrectly:
            - Tell them how many letters they have correct but in the wrong place
            - Tell them how many letters they have correct and in the correct place
            - Prompt them for another word and repeat this process
        - User guesses correctly:
            - Congratulate them and prompt them to either play again or exit 
