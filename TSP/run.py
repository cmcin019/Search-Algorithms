# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca


import argparse
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import os
from os import system

global max_it
max_it = 0

# Cost matrix for evaluation of solution fitness
def to_dict(nodes):
	# Create NxN distance matrix between cities 
	# We do not use this while exploring solution space 
	# This matrix is used as a look up when computing total cost
	# Instead of calculating the nome every time
	node_dict = {n+1:{_n+1:{} for _n in range(len(nodes))} for n in range(len(nodes))}
	node_coord_dict = {}
	list_i_1, list_i_2 = [], []
	for i in range(len(nodes)): 
		# City coord
		_, i_1, i_2 = nodes[i].split()

		# Add coord to list for visulization 
		list_i_1.append(i_1)
		list_i_2.append(i_2)

		# Add coord to coord dictionrary
		node_coord_dict[i+1] = (i_1, i_2)

		# Convert to string to float values 
		n_1 = np.array((float(i_1), float(i_2)))

		# For every node that comes after current node
		for j in range(i, len(nodes)):

			# Get second city coord 
			_, j_1, j_2  = nodes[j].split()

			# Convert to string to float values 
			n_2 = np.array((float(j_1), float(j_2)))

			# Calculate distance between cities
			dist = np.linalg.norm(n_1 - n_2)

			# Add distance to distance dictionary (first city)
			node_dict[i+1][j+1] = dist
			node_dict[i+1] = node_dict[i+1]

			# Add distance to distance dictionary (second city)
			node_dict[j+1][i+1] = dist
			node_dict[j+1] = node_dict[j+1]

	return node_dict, node_coord_dict, list_i_1, list_i_2

# Calculate cost of a given path
def get_cost(dist_dict, state):
	# Initial cost 
	cost = 0

	# Every element in state 
	for i in range(len(state)-1):  

		# Distance between current and next city
		cost +=  dist_dict[state[i]][state[i+1]]

	return cost

# Random start
def random_search(dist_dict, args, ax, i=1):
	plot_costs = []
	plot_it = []

	# Sample random state from dictionary
	state = list(sample(list(dist_dict.keys()), len(dist_dict)))

	# Get cost of random state
	cost = get_cost(dist_dict, state)

	# If multiple iterations are performed
	for it in range(i-1):
		if it % 10000 == 0: # Print every 10000 
			# Clear terminal 
			system('cls' if os.name == 'nt' else 'clear')
			# Make user contimplate whether running the script is worth it or not
			print('And now we play the waiting game \n')
			# Provide information about current step
			print(f'Random x {i}')
			print(f'Step: {it}')
			print(f'Path size: {cost}')
		
		# Sample new random state
		new_state = list(sample(list(dist_dict.keys()), len(dist_dict)))
		new_cost = get_cost(dist_dict, new_state)

		# Compare best random state cost with current cost
		if cost > new_cost :

			# Switch state and cost if current is better
			state, cost = list(new_state), new_cost	

		# Save best cost and iteration
		plot_costs.append(cost)
		plot_it.append(it)

	# Plot final results
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
	# Ploting 
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
	
				if it % 10000 == 0: # Print every 10000 
					# Clear terminal
					system('cls' if os.name == 'nt' else 'clear')
					# Let user know running the script may take a long long long time
					print('And now we play the waiting game \n')
					# Provide information about current step
					print(' '.join(name.split()[0:2]+['Rx4000' if '4000' in name else 'Rx1']))
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

					# Save results to plot 
					plot_costs.append(best_cost)
					plot_it.append(it)
					
					# Break out of loop after finding better state
					break
			
			# Break if better solution found and only doing first ascent
			if flag and not (aggressive or random_ascent):
				break
				
		# Chosing from neighbour states with lower costs 
		if flag and (aggressive or random_ascent):
			if aggressive: # Steepest ascent

				# Find minimum cost path
				best_state, best_cost = min(r_ascent_batch, key = lambda t: t[1])

				# Save results to plot 
				plot_costs.append(best_cost)
				plot_it.append(it)

			else: # Random ascent

				# Find random path and designated cost
				best_state, best_cost = sample(r_ascent_batch, 1)[0]

				# Save results to plot 
				plot_costs.append(best_cost)
				plot_it.append(it)
	
	# Save max iteration for final random search
	if it > max_it: 
		max_it = it

	# Plot results
	if args.l:
		ax.plot(plot_it, plot_costs, label=' '.join(name.split()[0:2]+[' Rx4000' if '4000' in name else 'Rx1']))
		ax.set(xlabel='Iteration', ylabel='Length')

	return best_state, best_cost

# Run first ascent hill climbing search using local search function 
def first_ascent_hill_climbing_search(dist_dict, state, cost, args, ax, name):
	# Not aggressive nor random
	# Chose first better solution
	return local_search(dist_dict, state, cost, args, ax, name, aggressive=False, random_ascent=False)

# Run steepest ascent hill climbing search using local search function 
def steepest_ascent_hill_climbing_search(dist_dict, state, cost, args, ax, name):
	# Search through all neighbour solutions 
	# Pick best
	return local_search(dist_dict, state, cost, args, ax, name, aggressive=True, random_ascent=False)
	
# Run random ascent hill climbing search using local search function 
def random_ascent_hill_climbing_search(dist_dict, state, cost, args, ax, name):
	# Search through all neighbour solutions 
	# Pick one at random
	return local_search(dist_dict, state, cost, args, ax, name, aggressive=False, random_ascent=True)

# Run search algorithm
def run(dist_dict, state, cost, search, name, args, node_coord_dict, list_i_1, list_i_2, ax):

	# Call search algorithm 
	state, cost = search(dist_dict, state, cost, args, ax, name)

	# Plot path found
	if args.s:

		# Plot cities
		plt.scatter(list_i_1, list_i_2)

		# Plot path between cities
		plt.plot(list(map(lambda x: node_coord_dict[int(x)][0], state)), list(map(lambda x: node_coord_dict[int(x)][1], state)))
		plt.show()

	return state, cost
	
def main() -> None:
	# Max iterations from algorithms for final random search
	global max_it
	# Arguments and values 
	parser = argparse.ArgumentParser()
	
	# Chose path to instance
	parser.add_argument("-path", help="Enter path to the instance file (include tsp)", type=str, default="Instances/a280.tsp")
	parser.add_argument("-alg", help="Enter number for alg \n-1: All (Default)\n0: First ascent from rand\n1: First ascent from randx4000\n2: Steepest ascent from rand\n3: Steepest ascent from randx4000\n4: Random ascent from rand\n5: Random ascent from randx4000", type=int, default=-1)
	parser.add_argument("-s", help="Plot paths found", action="store_true", default=False)
	parser.add_argument("-l", help="Plot path length", action="store_true", default=False)

	# Parse agruments
	args = parser.parse_args()
	
	# Read file
	# f = open(args.path)
	path = args.path 
	f = open(path)
	lines = f.readlines()
	f.close()

	# Find first city node
	lines = lines[lines.index('NODE_COORD_SECTION\n')+1:lines.index('EOF\n')]

	# Create dictionary
	dist_dict, node_coord_dict, list_i_1, list_i_2 = to_dict(lines)

	# Plot cities
	if args.s:
		plt.scatter(list_i_1, list_i_2)
		plt.show()

	fig, ax = plt.subplots()
	
	# Find random state
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

	# Testing First hill ascent from random state
	if args.alg == 0 or args.alg == -1:
		state, cost = run(dist_dict, random_state, random_cost, first_ascent_hill_climbing_search, 'First hill climbing search cost from random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)
		f = open("solutions/solution_0.csv", "w")
		for x in state:
			f.write(str(x)+'\n')
		f.close()

	# Testing First hill ascent from best random state of 40000
	if args.alg == 1 or args.alg == -1:
		state, cost = run(dist_dict, iter_random_state, iter_random_cost, first_ascent_hill_climbing_search, 'First hill climbing search cost from best out of 4000 random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)
		f = open("solutions/solution_1.csv", "w")
		for x in state:
			f.write(str(x)+'\n')
		f.close()


	# Testing Steepest hill ascent from random state
	if args.alg == 2 or args.alg == -1:
		state, cost = run(dist_dict, random_state, random_cost, steepest_ascent_hill_climbing_search, 'Steepest hill climbing search cost from random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)
		f = open("solutions/solution_2.csv", "w")
		for x in state:
			f.write(str(x)+'\n')
		f.close()

	# Testing Steepest hill ascent from best random state of 40000
	if args.alg == 3 or args.alg == -1:
		state, cost = run(dist_dict, iter_random_state, iter_random_cost, steepest_ascent_hill_climbing_search, 'Steepest hill climbing search cost from best out of 4000 random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)
		f = open("solutions/solution_3.csv", "w")
		for x in state:
			f.write(str(x)+'\n')
		f.close()


	# Testing Random hill ascent from random state
	if args.alg == 4 or args.alg == -1:
		state, cost = run(dist_dict, random_state, random_cost, random_ascent_hill_climbing_search, 'Random hill climbing search cost from random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)
		f = open("solutions/solution_4.csv", "w")
		for x in state:
			f.write(str(x)+'\n')
		f.close()

	# Testing Random hill ascent from best random state of 40000
	if args.alg == 5 or args.alg == -1:
		state, cost = run(dist_dict, iter_random_state, iter_random_cost, random_ascent_hill_climbing_search, 'Random hill climbing search cost from best out of 4000 random start: ', args, node_coord_dict, list_i_1, list_i_2, ax)
		f = open("solutions/solution_5.csv", "w")
		for x in state:
			f.write(str(x)+'\n')
		f.close()

	# Best of max iterations random states
	iter_random_state_max_it, iter_random_cost_max_it = random_search(dist_dict, args, ax, i=max_it)
	f = open("solutions/solution_rand.csv", "w")
	for x in iter_random_state_max_it:
		f.write(str(x)+'\n')
	f.close()

	if args.l:
		plt.legend()
		plt.show()

	# Save results
	f = open("solution.csv", "w")
	for x in state:
		f.write(str(x)+'\n')
	f.close()

	# Clear and print final cost
	system('cls' if os.name == 'nt' else 'clear')
	print(cost)
	

if __name__ == "__main__":
	main()
	# print("End")
