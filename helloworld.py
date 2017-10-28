print('Hello world')

e1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
e2 = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')]

# Print out elements in a list of strings
for char in e1:
	print('Next character: ' + char) 


print('\n\n')
	
# Print out elements in list of tuples each field seperately 
for (number, letter) in e2:
	print('The number is: ' + str(number))
	print('The letter is: ' + letter)
	print('--------')