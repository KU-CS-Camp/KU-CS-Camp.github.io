---
title: Python Review Exercises
layout: default
filename: python-review-exercises.md
--- 

## Python Review Exercises

### Resources
- [Terminal Commands](https://www.guru99.com/linux-commands-cheat-sheet.html)
- [Python Reference](https://www.w3schools.com/python/)
- [NumPy Reference](https://www.w3schools.com/python/numpy/numpy_intro.asp)
***
### Exercise 1 - Hello World

Every programmer's first program is Hello World. In this exercise, we will go through the process of creating a file, writing a basic Python print command, and use a terminal to execute the program.

1. Create a new folder named ‘python-review’
2. Open the Atom application
3. Open your new ‘Python Review’ folder in Atom
4. Create a new file named ‘helloworld.py’
5. Type the following code in the file:
```
print('Hello, World!')
```
6. Open a Terminal window
7. Navigate to your folder using cd (cd Desktop/pythonreview)
8. Type the command ‘python helloworld.py’ and hit Enter
***
### Exercise 2 - Grade Checker

Pretend that we have a list of grades from a hypothetical CS course. We can use this list of grades to explore lists, if statements, and for loops.

1. Create a new file named ‘grades.py’
2. Generate a list of ~10 fake grades (your choice)
3. Create a function for checking grades
4. Loop through the grades and print out ‘You passed!’ when the grade is above a 70 otherwise print ‘Sorry, you did not pass’
5. Call the function at the bottom of the file
***
### Exercise 3 - NumPy Arrays

Now, we will practice using a Python library called NumPy. This will be helpful in future projects and give you experience looking at library documentation.

1. Create a new file called ‘numpy_practice.py’
2. Import the NumPy library:
```
import numpy as np
```
Using ‘as np’ means that when you reference the library you can just use np rather than type numpy every time. For example, call np.function(array)

3. Create a list of random integers
4. Print out the results of using NumPy to find the following (For syntax: [NumPy Reference](https://www.w3schools.com/python/numpy/numpy_intro.asp)):
    - Size
    - Minimum and maximum (amax, amin)
    - Mean
    - Shape (dimensions of the array)
    - Sorted array
