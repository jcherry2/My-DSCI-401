# This is a test driver for our functions defined in basic_functions.py

# Import this module and name as bs
import basic_functions as bs

#print(bs.add_2(3, 5))
#print(bs.add_2(4, 6))
#print(bs.add_2(7, 8))


#print(bs.my_range(1,50, 3))
#print(bs.my_range(1, 50, by=4)) 

#print(bs.my_range(1, 11, 3))
#print(bs.histogram(['a', 'x', 2, 'x', 3, 2]))
#print(bs.histogram(['a', 'b', 'b', 'c', 'e', 'a', 'd']))

#bs.print_triangle(3)
#bs.print_triangle(5)
#bs.print_triangle(5, full=True)


# Test the word_count function.
print(bs.word_counts('./data/sample_text.txt', case_sensitive = False))

print(bs.my_max([1, 6, 9, 4, 6]))

# Test the fzip function 
print(bs.fzip(lambda x, y: x + y, [1,2,3], [4,5,6]))
print(bs.fzip(max, [1,2,3], [4,5,6], [7,8,9]))

print(bs.sum_range(1,10))
print(bs.sum_range(1,100))

print(bs.rrev([1,2,3,4,5]))

print(bs.fib(1,1,8))

print(bs.mfib(1,1,100))

print(bs.kcomb([1, 2, 3, 4], 2)

# testing pipe ---
f1 = lambda x: x + 3
f2 = lambda x: x + x
f3 = lambda x: x / 2.3
f4 = lambda x: x ** 0.5

my_pipe = bs.pipe(f1, f2, f3, f4)

print(list(map(my_pipe, range(1, 21))))

import cherry_assignment_1 as a1
print(a1.flatten([[1,2,3], [[4],[5]], 6,7,8]))

