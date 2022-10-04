# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114

import argparse
from asyncio import current_task
from math import cos
# from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from random import randint, sample
from operator import itemgetter
import os
from os import stat, system
from timeit import default_timer
# from tqdm import tqdm

# Cost matrix for evaluation of solution fitness
def to_dict(nodes):
	# Create NxN distance matrix between cities 
	# We do not use this while exploring solution space 
	# This matrix is used as a look up when computing total cost
	node_dict = {n+1:{_n+1:{} for _n in range(len(nodes))} for n in range(len(nodes))}
	node_coord_dict = {}
	list_i_1, list_i_2 = [], []
	for i in range(len(nodes)):
		_, i_1, i_2 = nodes[i].split()
		list_i_1.append(i_1)
		list_i_2.append(i_2)
		node_coord_dict[i+1] = (i_1, i_2)
		n_1 = np.array((float(i_1), float(i_2)))
		for j in range(i, len(nodes)):
			_, j_1, j_2  = nodes[j].split()

			n_2 = np.array((float(j_1), float(j_2)))
			dist = np.linalg.norm(n_1 - n_2)

			node_dict[i+1][j+1] = dist
			node_dict[i+1] = node_dict[i+1]
			
			node_dict[j+1][i+1] = dist
			node_dict[j+1] = node_dict[j+1]
	return node_dict, node_coord_dict, list_i_1, list_i_2

# Calculate cost of a given path
def get_cost(dist_dict, state):
	cost = 0
	for i in range(len(state)-1):
		cost +=  dist_dict[state[i]][state[i+1]]
	return cost

# Random start
def random_search(dist_dict, i=1):
	state = sample(list(dist_dict.keys()), len(dist_dict))
	cost = get_cost(dist_dict, state)
	for _ in range(i-1):
		new_state = sample(list(dist_dict.keys()), len(dist_dict))
		new_cost = get_cost(dist_dict, new_state)
		if cost > new_cost :
			state, cost = new_state, new_cost	
	return state, cost

# Look through all neighbour solutions
def local_search(dist_dict, state, cost, aggressive=False, random_ascent=False, random_restart=0, T=0):
	# Initial best state and cost
	current_state = state
	best_state = state
	best_cost = cost
	
	# Continue while better states are found
	flag = True 
	it = 0
	while flag :

		# List of solutions
		r_ascent_batch = []

		# False if no better solutions are found
		flag = False
		
		# Look through every neighbour
		for i in range(len(state)): # Fist neighbour to swap
			for j in range(i+1, len(state)): # Second neighbour to swap
				# Number of neighbours swaped
				it +=1
				
				# Neighbour state
				current_state = [x for x in best_state]
				current_state[i], current_state[j] = best_state[j], best_state[i]

				# print(current_state == best_state, best_state[j], best_state[i], current_state[j], current_state[i], i, j)
				# Neighbour cost
				current_cost = get_cost(dist_dict, current_state)
				
				# Compare current best and neighbour cost
				if current_cost < best_cost:
				
					# Keep searching if neighbour cost is better
					flag = True
					
					# For random ascent hill climbing
					if random_ascent or aggressive:

						# Add neighbour to list of better states
						r_ascent_batch.append((current_state, current_cost))
						# print(current_state == best_state)

						
						# Do not update best state and cost
						continue
					
					# Update the best state and cost
					best_state = current_state
					best_cost = current_cost
					
					# Break out of loop after finding better state
					break
			
			# Break if better solution found and only doing first ascent
			if flag and not (aggressive or random_ascent):
				break
				
		# Chosing from neighbour states with lower costs 
		if flag and (aggressive or random_ascent):
			if aggressive:
				best_state, best_cost = min(r_ascent_batch, key = lambda t: t[1])
				# best_state = state
			else:
				best_state, best_cost = sample(r_ascent_batch, 1)[0]
				# best_state = state

	print('Steps:', it)
	
	# # Random restart
	# if random_restart > 0:
	# 	rand_state, rand_cost = random_search(dist_dict)
	# 	rand_state, rand_cost = local_search(dist_dict, rand_state, rand_cost, aggressive=aggressive, random_ascent=random_ascent, random_restart=random_restart-1)
	# 	if rand_cost < best_cost:
	# 		best_state, best_cost = rand_state, rand_cost

	# # Simulated annealing
	# if T > 0:
	# 	pass

	# Check if cost was properly calculated
	cost_ = 0
	cost_from_get = get_cost(dist_dict, best_state)
	for x in range(len(best_state)-1):
		cost_ += dist_dict[best_state[x]][best_state[x+1]]
	print(cost_, best_cost, cost_from_get, best_cost == cost_)


	return best_state, best_cost

def first_ascent_hill_climbing_search(dist_dict, state, cost):
	return local_search(dist_dict, state, cost, aggressive=False, random_ascent=False)

def steepest_ascent_hill_climbing_search(dist_dict, state, cost):
	return local_search(dist_dict, state, cost, aggressive=True, random_ascent=False)
	
def random_ascent_hill_climbing_search(dist_dict, state, cost):
	return local_search(dist_dict, state, cost, aggressive=False, random_ascent=True)


def run(dist_dict, state, cost, search, name, args, node_coord_dict, list_i_1, list_i_2):
	start = default_timer()
	state, cost = search(dist_dict, state, cost)
	stop = default_timer()

	# Check if cost was properly calculated
	cost_ = 0
	cost_from_get = get_cost(dist_dict, state)
	for x in range(len(state)-1):
		cost_ += dist_dict[state[x]][state[x+1]]
	print(cost_, cost, cost_from_get, cost == cost_)

	print('Exec time:', stop - start)
	print(name)
	print(cost,'\n')
	if args.s:
		plt.scatter(list_i_1, list_i_2)
		plt.plot(list(map(lambda x: node_coord_dict[int(x)][0], state)), list(map(lambda x: node_coord_dict[int(x)][1], state)))
		plt.show()
	return state, cost
	

def main() -> None:
	
	# Arguments and values 
	parser = argparse.ArgumentParser()
	
	# Chose path to instance
	# parser.add_argument("path", help="Enter path tho the instance file (include tsp)", type=str)
	parser.add_argument("-s", help="Plot data and paths found", action="store_true", default=False)
	
	# Parse agruments
	args = parser.parse_args()
	
	# Read file
	# f = open(args.path)
	# path = args.path 
	path = "Instances/a280.tsp"
	f = open(path)
	lines = f.readlines()
	f.close()

	
	lines = lines[lines.index('NODE_COORD_SECTION\n')+1:lines.index('EOF\n')]
	# print(lines[0])

	# return 0
	dist_dict, node_coord_dict, list_i_1, list_i_2 = to_dict(lines)
	if args.s:
		plt.scatter(list_i_1, list_i_2)
		#plt.plot([1,4,5], [2,2,3])
		plt.show()
	
	# Random state
	start = default_timer()
	random_state, random_cost = random_search(dist_dict)
	stop = default_timer()
	print('Exec time:', stop - start)
	print('First Random search cost: ')
	print(random_cost, '\n')
	
	if args.s:
		plt.scatter(list_i_1, list_i_2)
		plt.plot(list(map(lambda x: node_coord_dict[int(x)][0], random_state)), list(map(lambda x: node_coord_dict[int(x)][1], random_state)))
		plt.show()

	# Best of 100 random states
	start = default_timer()
	iter_random_state, iter_random_cost = random_search(dist_dict, i=100)
	stop = default_timer()
	#print(cost,'\n',new_state)
	print('Exec time:', stop - start)
	print('Best of 100 iteration random search cost: ')
	print(iter_random_cost,'\n')

	if args.s:
		plt.scatter(list_i_1, list_i_2)
		plt.plot(list(map(lambda x: node_coord_dict[int(x)][0], iter_random_state)), list(map(lambda x: node_coord_dict[int(x)][1], iter_random_state)))
		plt.show()


	# state, cost = run(dist_dict, random_state, random_cost, first_ascent_hill_climbing_search, 'First hill climbing search cost from random start: ', args, node_coord_dict, list_i_1, list_i_2)
	# state, cost = run(dist_dict, iter_random_state, iter_random_cost, first_ascent_hill_climbing_search, 'First hill climbing search cost from best out of 100 random start: ', args, node_coord_dict, list_i_1, list_i_2)

	state, cost = run(dist_dict, random_state, random_cost, steepest_ascent_hill_climbing_search, 'Steepest hill climbing search cost from random start: ', args, node_coord_dict, list_i_1, list_i_2)
	state, cost = run(dist_dict, iter_random_state, iter_random_cost, steepest_ascent_hill_climbing_search, 'Steepest hill climbing search cost from best out of 100 random start: ', args, node_coord_dict, list_i_1, list_i_2)

	state, cost = run(dist_dict, random_state, random_cost, random_ascent_hill_climbing_search, 'Random hill climbing search cost from random start: ', args, node_coord_dict, list_i_1, list_i_2)
	state, cost = run(dist_dict, iter_random_state, iter_random_cost, random_ascent_hill_climbing_search, 'Random hill climbing search cost from best out of 100 random start: ', args, node_coord_dict, list_i_1, list_i_2)

	f = open("solution.csv", "w")
	for x in state:
		f.write(str(x)+'\n')
	f.close()

	# system('cls' if os.name == 'nt' else 'clear')
	# print(cost)
	

if __name__ == "__main__":
	print()
	main()
	print("End")
	
	
