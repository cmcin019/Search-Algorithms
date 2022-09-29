# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114

import argparse
# from typing import Dict
import numpy as np
from random import randint, sample
from operator import itemgetter
import os
from os import system
from timeit import default_timer
# from tqdm import tqdm

# Cost matrix for evaluation of solution fitness
def to_dict(nodes):
	
	node_dict = {n+1:{_n+1:{} for _n in range(len(nodes))} for n in range(len(nodes))}
	
	for i in range(len(nodes)):
		_, i_1, i_2 = nodes[i].split()
		n_1 = np.array((float(i_1), float(i_2)))

		for j in range(i, len(nodes)):
			_, j_1, j_2  = nodes[j].split()

			n_2 = np.array((float(j_1), float(j_2)))
			dist = np.linalg.norm(n_1 - n_2)

			node_dict[i+1][j+1] = dist
			node_dict[i+1] = node_dict[i+1]
			
			node_dict[j+1][i+1] = dist
			node_dict[j+1] = node_dict[j+1]
	
	return node_dict

# Calculate cost of a given path
def get_cost(dist_dict, state):
	cost = [0]
	for i in range(len(state)-1):
		cost.append(cost[-1] + dist_dict[state[i]][state[i+1]])
	return cost

# Random start
def random_search(dist_dict, i=1):
	state = sample(list(dist_dict.keys()), len(dist_dict))
	cost = get_cost(dist_dict, state)
	for _ in range(i-1):
		new_state = sample(list(dist_dict.keys()), len(dist_dict))
		new_cost = get_cost(dist_dict, new_state)
		if cost[-1] > new_cost[-1] :
			state, cost = new_state, new_cost	
	return state, cost

# Look through all neighbour solutions
def local_search(dist_dict, state, cost, aggressive=False, random_ascent=False, random_restart=0, T=0):
	# Initial best state and cost
	best_state = state
	best_cost = cost
	
	# Continue while better states are found
	flag = True 
	it = 0
	while flag :
		r_ascent_batch = []
		costs = []
		flag = False
		
		# Look through every neighbour
		for i in range(len(state)):
			for j in range(i, len(state)):
				it +=1
				
				# Neighbour state
				state[i], state[j] = state[j], state[i]
				
				# Neighbour cost
				cost = get_cost(dist_dict, state)
				
				# Compare current best and neighbour cost
				if cost[-1] < best_cost[-1]:
				
					# Keep searching if neighbour cost is better
					flag = True
					
					# For random ascent hill climbing
					if random_ascent or aggressive:
						
						# Add neighbour to list of better states
						r_ascent_batch.append((state, cost))
						costs.append(cost[-1])
						
						# Reset state
						state = best_state
						
						# Do not update best state and cost
						continue
					
					# Update the best state and cost
					best_state = state
					best_cost = cost
					
					# Break out of loop after finding better state
					break
				
				# Neighbour cost is not better 
				else :
					# Reset state
					state = best_state
			
			# Break if better solution found and only doing first ascent
			if flag and not (aggressive or random_ascent):
				break
				
		# Chosing from neighbour states with lower costs 
		if (flag and (aggressive or random_ascent)):
			if aggressive:
				best_state, best_cost = r_ascent_batch[costs.index(min(costs))]
			else:
				best_state, best_cost = sample(r_ascent_batch, 1)[0]
				
	print('Steps:', it)
	
	# # Random restart
	# if random_restart > 0:
	# 	rand_state, rand_cost = random_search(dist_dict)
	# 	rand_state, rand_cost = local_search(dist_dict, rand_state, rand_cost, aggressive=aggressive, random_ascent=random_ascent, random_restart=random_restart-1)
	# 	if rand_cost[-1] < best_cost[-1]:
	# 		best_state, best_cost = rand_state, rand_cost

	# # Simulated annealing
	# if T > 0:
	# 	pass
	cost_ = 0
	for x in range(len(state)-1):
		cost_ += dist_dict[state[x]][state[x+1]]
	print(cost_ == cost[-1])
	print(cost[-1], best_cost[-1])
	return best_state, best_cost

def first_ascent_hill_climbing_search(dist_dict, state, cost):
	return local_search(dist_dict, state, cost, aggressive=False, random_ascent=False)

def steepest_ascent_hill_climbing_search(dist_dict, state, cost):
	return local_search(dist_dict, state, cost, aggressive=True, random_ascent=False)
	
def random_ascent_hill_climbing_search(dist_dict, state, cost):
	return local_search(dist_dict, state, cost, aggressive=False, random_ascent=True)


def run(dist_dict, state, cost, search, name):
	start = default_timer()
	state, cost = search(dist_dict, state, cost)
	stop = default_timer()
	#print(cost[-1],'\n',new_state)

	# Check if cost was properly calculated
	cost_ = 0
	for x in range(len(state)-1):
		cost_ += dist_dict[state[x]][state[x+1]]
	print(cost_ == cost[-1])

	print('Exec time:', stop - start)
	print(name)
	print(cost[-1],'\n')
	return state, cost
	

def main() -> None:
	
	# Arguments and values 
	parser = argparse.ArgumentParser()
	
	# Chose path to instance
	parser.add_argument("path", help="Enter path tho the instance file (include tsp)", type=str)
	
	# Parse agruments
	args = parser.parse_args()
	
	# Read file
	f = open(args.path)
	lines = f.readlines()
	f.close()
	
	lines = lines[6:-1][:100]
	dist_dict = to_dict(lines)
	
	# Random state
	start = default_timer()
	random_state, random_cost = random_search(dist_dict)
	stop = default_timer()
	print('Exec time:', stop - start)
	print('First Random search cost: ')
	print(random_cost[-1], '\n')
	
	# Best of 100 random states
	start = default_timer()
	iter_random_state, iter_random_cost = random_search(dist_dict, i=100)
	stop = default_timer()
	#print(cost[-1],'\n',new_state)
	print('Exec time:', stop - start)
	print('Best of 100 iteration random search cost: ')
	print(iter_random_cost[-1],'\n')

	state, cost = run(dist_dict, random_state, random_cost, first_ascent_hill_climbing_search, 'First hill climbing search cost from random start: ')
	state, cost = run(dist_dict, iter_random_state, iter_random_cost, first_ascent_hill_climbing_search, 'First hill climbing search cost from best out of 100 random start: ')

	state, cost = run(dist_dict, random_state, random_cost, steepest_ascent_hill_climbing_search, 'Steepest hill climbing search cost from random start: ')
	state, cost = run(dist_dict, iter_random_state, iter_random_cost, steepest_ascent_hill_climbing_search, 'Steepest hill climbing search cost from best out of 100 random start: ')

	state, cost = run(dist_dict, random_state, random_cost, random_ascent_hill_climbing_search, 'Random hill climbing search cost from random start: ')
	state, cost = run(dist_dict, iter_random_state, iter_random_cost, random_ascent_hill_climbing_search, 'Random hill climbing search cost from best out of 100 random start: ')

	f = open("solution.csv", "w")
	for x in state:
		f.write(str(x)+'\n')
	f.close()

	# system('cls' if os.name == 'nt' else 'clear')
	print(cost[-1])
	

if __name__ == "__main__":
	main()
	print("End")

