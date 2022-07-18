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
2. Open the Atom application by clicking 'Menu' in the bottom left corner and searching for it
3. Open your new ‘Python Review’ folder in Atom (File > Open)
4. Create a new file named ‘helloworld.py’ (File > Create New File)
5. Type the following code in the file:
```
print('Hello, World!')
```
6. Open a Terminal window (you can also search for this in the bottom left corner)
7. Use the command ls to find the folder to navigate into
8. Navigate to your folder using cd (cd Desktop/pythonreview)
9. Type the command ‘python helloworld.py’ and hit Enter

***
### Exercise 2 - Grade Checker

Pretend that we have a list of grades from a hypothetical CS course. We can use this list of grades to explore lists, if statements, and for loops.

1. Create a new file named ‘grades.py’
2. Create a variable equal to an array of ~10 fake grades (integer numbers). For example, 66,89, 99, 70
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

3. Create a variable equal to an array of random integers. For example, 77, 23, 1, 34
4. Print out the results of using NumPy to find the following (For syntax: [NumPy Reference](https://numpy.org/doc/stable/reference/routines.math.html):
    - Size
    - Minimum and maximum (amax, amin)
    - Mean
    - Shape (dimensions of the array)
    - Sorted array

***
### Exercise 4 - Restaurant Menu

Welcome to your very own restaurant! At your restaurant, you have 3 options: an appetizer, entree, and dessert.  You can choose what specific food you want these to be and how you will price them. Now, you need to create an interactive ordering program for your customers.

Your program will display text in the terminal and get input from the terminal. You will need to use [Python Input/Output](https://www.geeksforgeeks.org/taking-input-from-console-in-python/) to get input from the terminal. It will look similar to the following:
```
print('Welcome')
num = int(input('Enter number: '))
```

The program will follow these steps:
1. You will welcome the customer then lead them through your food options one by one asking if they would like that item. For each item (appetizer->entree->dessert):
- Ask if the customer wants the item and accept a yes/no answer. For yes they can type 'y' or 'Y', and for no they can type 'n' or 'N' 
- If they want an item, ask how many
2. Ask for the customer's age in order to apply a discount if applicable
- 65+ get a 10% discount
- <5 get all desserts free
3. Finally, display their receipt
- Calculate/display the following:
    - Cost per item (before any discounts)
    - Subtotal (before any discounts)
    - Tax amount (tax is 5%)
    - Discount amount
    - Grand total

***

[Here](https://www.geeksforgeeks.org/python-projects-beginner-to-advanced/) are more Python project ideas if you finish early!

