# Josiah Cherry

# 1
# flatten function takes a list and puts all items into a single list. 

def flatten(List, x = None):
    if x is None:
        x = []

    for i in List:
        if isinstance(i, list):
            flatten(i, x)
        else:
            x.append(i)
    return x
	


# 2
# set of all possible subsets of a set	
from functools import reduce
def powerset(List):
    return reduce(lambda z, x: z + [y + [x] for y in z], List, [[]])
	

# 3
# produces all permutations of a list
def all_perms(List):
    # for no elements in list
    if len(List) == 0:
        return []
		
    # If there is only one element in list
    if len(List) == 1:
        return [List]
 
	# more than 1 
    perm = [] 
    for i in range(len(List)):
       x = List[i]
       newList = List[:i] + List[i+1:]
       for y in all_perms(newList):
           perm.append([x] + y)
    return perm

				
# 4
# Prints a number spiral given a number.

def Spiral(end):
	spiral = []
	num = end
	num2 = (num // 2 + 1)
	num_count = num - 1

	for i in range(num):
		s = []
		if (i < num2):
			for k in range (i):
				if i == 0:
					break
				else:
					s.append(spiral[i-1][k] - 1)

		for j in range (num):
			
			s.append((num * num) - (num_count))
			num_count -= 1

		if (i < num2):
			for k in range (i, 0, -1):
				if i == 0:
					break
				if k == i:
					s.append(s[-1] +1)
				else:
					s.append(spiral[i-1][-k] + 1)

		if (i >= num2):
			for k in range(i + num_count):
				s.append(spiral[i-1][k] - 1 )
			
		if (i >= num2):
			for x in range(1, num -1, -1):
				s.append(num_count * num_count + x)

		if (i >= num2):
			for k in range(i + num_count, 0, -1):
				s.append(spiral[i-1][-k] + 1)

		num -= 2
		num_count = num - 1
		spiral.append(s)

	return(spiral)	