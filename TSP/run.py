# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114

import argparse
from cProfile import label
# from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from random import randint, sample
import os
from os import stat, system
from timeit import default_timer
# from tqdm import tqdm
global max_it
max_it = 0
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
def random_search(dist_dict, args, ax, i=1):
	plot_costs = []
	plot_it = []
	state = list(sample(list(dist_dict.keys()), len(dist_dict)))
	cost = get_cost(dist_dict, state)
	for it in range(i-1):
		if it % 10000 == 0:
			system('cls' if os.name == 'nt' else 'clear')
			print(f'Random x {i}')
			print(f'Step: {it}')
			print(f'Path size: {cost}')
		new_state = list(sample(list(dist_dict.keys()), len(dist_dict)))
		new_cost = get_cost(dist_dict, new_state)
		if cost > new_cost :
			state, cost = list(new_state), new_cost	
		plot_costs.append(cost)
		plot_it.append(it)
	if args.l and i != 1:
		ax.plot(plot_it, plot_costs, label=f'Random x {i}')
		ax.set(xlabel='Iteration', ylabel='Length')
	return state, cost

# Look through all neighbour solutions
def local_search(dist_dict, state, cost, args, ax, name, aggressive=False, random_ascent=False):
	# Initial best state and cost
	current_state = state
	best_state = state
	best_cost = cost
	plot_costs = []
	plot_it = []
	global max_it

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
	
				if it % 10000 == 0:
					system('cls' if os.name == 'nt' else 'clear')
					print('And now we play the waiting game \n')
					print(' '.join(name.split()[0:2]+['Rx4000' if '4' in name else 'Rx1']))
					print(f'Step: {it}')
					print(f'Path size: {best_cost}')

				# Neighbour state
				# current_state = [x for x in best_state]
				current_state = list(best_state)
				current_state[i], current_state[j] = best_state[j], best_state[i]

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
						
						# Do not update best state and cost
						continue
					
					# Update the best state and cost
					best_state = current_state
					best_cost = current_cost
					plot_costs.append(best_cost)
					plot_it.append(it)
					
					# Break out of loop after finding better state
					break
			
			# Break if better solution found and only doing first ascent
			if flag and not (aggressive or random_ascent):
				break
				
		# Chosing from neighbour states with lower costs 
		if flag and (aggressive or random_ascent):
			if aggressive:
				best_state, best_cost = min(r_ascent_batch, key = lambda t: t[1])
				plot_costs.append(best_cost)
				plot_it.append(it)
			else:
				best_state, best_cost = sample(r_ascent_batch, 1)[0]
				plot_costs.append(best_cost)
				plot_it.append(it)
	
	if it > max_it:
		max_it = it

	if args.l:
		ax.plot(plot_it, plot_costs, label=' '.join(name.split()[0:2]+[' Rx4000' if '4' in name else 'Rx1']))
		ax.set(xlabel='Iteration', ylabel='Length')

	return best_state, best_cost

def first_ascent_hill_climbing_search(dist_dict, state, cost, args, ax, name):
	return local_search(dist_dict, state, cost, args, ax, name, aggressive=False, random_ascent=False)

def steepest_ascent_hill_climbing_search(dist_dict, state, cost, args, ax, name):
	return local_search(dist_dict, state, cost, args, ax, name, aggressive=True, random_ascent=False)
	
def random_ascent_hill_climbing_search(dist_dict, state, cost, args, ax, name):
	return local_search(dist_dict, state, cost, args, ax, name, aggressive=False, random_ascent=True)


def run(dist_dict, state, cost, search, name, args, node_coord_dict, list_i_1, list_i_2, ax):
	state, cost = search(dist_dict, state, cost, args, ax, name)
	if args.s:
		plt.scatter(list_i_1, list_i_2)
		plt.plot(list(map(lambda x: node_coord_dict[int(x)][0], state)), list(map(lambda x: node_coord_dict[int(x)][1], state)))
		plt.show()
	return state, cost
	

def main() -> None:
	global max_it
	# Arguments and values 
	parser = argparse.ArgumentParser()
	
	# Chose path to instance
	parser.add_argument("-path", help="Enter path tho the instance file (include tsp)", type=str, default="Instances/a280.tsp")
	parser.add_argument("-alg", help="Enter number for alg \n-1: All (Default)\n0: First ascent from rand\n1: First ascent from randx4000\n2: Steepest ascent from rand\n3: Steepest ascent from randx4000\n4: Random ascent from rand\n5: Random ascent from randx4000", type=int, default=-1)
	parser.add_argument("-s", help="Plot Ppaths found", action="store_true", default=False)
	parser.add_argument("-l", help="Plot path length", action="store_true", default=False)

	
	# Parse agruments
	args = parser.parse_args()
	
	# Read file
	f = open(args.path)
	path = args.path 
	# path = "Instances/a280.tsp"
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

	fig, ax = plt.subplots()
	
	# Random state
	random_state, random_cost = random_search(dist_dict, args, ax)
	
	if args.s:
		plt.scatter(list_i_1, list_i_2)
		plt.plot(list(map(lambda x: node_coord_dict[int(x)][0], random_state)), list(map(lambda x: node_coord_dict[int(x)][1], random_state)))
		plt.show()

	# Best of 4000 random states
	iter_random_state, iter_random_cost = random_search(dist_dict, args, ax, i=4000)

	if args.s:
		plt.scatter(list_i_1, list_i_2)
		plt.plot(list(map(lambda x: node_coord_dict[int(x)][0], iter_random_state)), list(map(lambda x: node_coord_dict[int(x)][1], iter_random_state)))
		plt.show()

	if args.alg == 0 or args.alg == -1:
		state, cost = run(dist_dict, random_state, random_cost, first_ascent_hill_climbing_search, 'First hill climbing search cost from random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)
	if args.alg == 1 or args.alg == -1:
		state, cost = run(dist_dict, iter_random_state, iter_random_cost, first_ascent_hill_climbing_search, 'First hill climbing search cost from best out of 4000 random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)

	if args.alg == 2 or args.alg == -1:
		state, cost = run(dist_dict, random_state, random_cost, steepest_ascent_hill_climbing_search, 'Steepest hill climbing search cost from random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)
	if args.alg == 3 or args.alg == -1:
		state, cost = run(dist_dict, iter_random_state, iter_random_cost, steepest_ascent_hill_climbing_search, 'Steepest hill climbing search cost from best out of 4000 random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)

	if args.alg == 4 or args.alg == -1:
		state, cost = run(dist_dict, random_state, random_cost, random_ascent_hill_climbing_search, 'Random hill climbing search cost from random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)
	if args.alg == 5 or args.alg == -1:
		state, cost = run(dist_dict, iter_random_state, iter_random_cost, random_ascent_hill_climbing_search, 'Random hill climbing search cost from best out of 4000 random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)

	# Best of max iterations random states
	iter_random_state_500000, iter_random_cost_500000 = random_search(dist_dict, args, ax, i=max_it)

	if args.l:
		plt.legend()
		plt.show()

	f = open("solution.csv", "w")
	for x in state:
		f.write(str(x)+'\n')
	f.close()

	system('cls' if os.name == 'nt' else 'clear')
	print(cost)
	

if __name__ == "__main__":
	print()
	main()
	# print("End")
