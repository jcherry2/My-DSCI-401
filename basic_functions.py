# These are some examples of python-style function definitions

# add two numbers
def add_2(x, y):
	return x + y
	
	
# Illustrate default arguments:
#def my_range(start, end, by=1):
	#return range(start, end, by)
	# Homework: rewrite this function to use a for loop rather than resorting to pythons range function. 
	
#def my_range(start, end, by=1):
	rng = []
	next_val = start
	while next_val <= end:
		rng.append(next_val)
		next_val += by
	return rng

# Print a triangle of the specified length/height
# If full == true, n is the height.	


#def print_triangle(side_lengths, full=False):
	line_number = 1
	while(line_number <= side_lengths):
		line = ""
		stars = 0
		while(stars < line_number):
			line = line + "*"
			stars += 1
		#print(line)
		line_number += 1
		
	if(full):
		line_number -= 2
		while(line_number >= 1):
			line = ""
			stars += 1
		#print(line)
		line_number -= 1
		
def print_triangle(n, full=False):
	pos = 1 
	while pos <= n:
		print('*' * n)
		pos += 1
	if full: 
		pos = n - 1 
		while pos >= 1:
			print('*' * n)
			pos -= 1

			
def histogram(items):
	d = {}
	for i in items:
		if not(i in d):
			d[i] = 0
		d[i] += 1
	return d
	
def word_counts(file_path, case_sensitive=True, punct = [':', '.', '-', '"', '?']):
	text = open(file_path, 'r').read()
	if not(case_sensitive):
		text = text.lower()
	# TODO: add code to count each punctuation character
	# for the time being, remove punctuation characters
	for p in punct:
		text = text.replace(p, ' ')
	words = text.split(' ')
	cleaned_words = []
	for w in words: 
		if len(w) > 0:
			cleaned_words.append(w.strip())
	return histogram(cleaned_words)
	
# returns the maximum number in elements list	
def my_max(elements):
	max = elements[0]
	for element in elements:
		if element > max:
			max = element
	return max

	#not finished
def variables_number_of_inputs(a, b, *rest):
	print('A is ' + str(a))
	print('B is ' + str(b))
	for e in rest:
		print('   Next Optional Input: ' + str(e))
		
# implement fzip - zip a set of lists and collapse resulting tuples

def fzip(f, *lists):
	return map(lambda tup: f(*tup), zip(*lists))

# a recursive definition of summing numbers over a range
def sum_range(a, b):
	if a == b: 
		return a 
	else: 
		return sum_range(a, b - 1) + b
# takes list and reverses it		
def rrev(list):
    return [list[-1]] + rrev(list[:-1]) if list else []
	
def fib1(first, second, n): 
	if n == 3:
		return first + second
	return fib1(second, first + second, n - 1)
# another version	
def fib(first, second, n):
	if n == 1:
		return first
	if n == 2:
		return second
	else:
		return fib(first, second, n - 1) + fib(first, second, n - 2)
# use memoization to aviod resolving sub-problems
def mfib(first, second, n, cache={}):
	if n == 1:
		return first
	if n == 2:
		return second
	elif n in cache:
		return cache[n]
	else:
		v = mfib(first, second, n - 1, cache) + mfib(first, second, n - 2, cache)
		cache[n] = v
		return v
	
# Finds all distinct combinations of k elements taken from elts.	
def kcomb(elts, k):
	if len(elts) == k:
		return [elts]
	if k == 1:
		return list(map(lambda x: [x], elts))
	else: 
		partials = kcomb(elts[1:], k - 1)
		return list(map(lambda x: [elts[0]] + x, partials)) + kcomb(elts[1:], k)
