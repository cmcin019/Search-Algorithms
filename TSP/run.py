# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114

import argparse
from typing import Dict
import numpy as np
from random import randint, sample
from operator import itemgetter
from tqdm import tqdm

def to_dict(nodes):
	
	node_dict = {n+1:{_n+1:{} for _n in range(len(nodes))} for n in range(len(nodes))}
	
	for i in range(len(nodes)):
		# print(nodes[i])
		try:
			_, i_1, i_2 = nodes[i].split()
			n_1 = np.array((float(i_1), float(i_2)))
		except:
			break
		
		for j in range(i, len(nodes)):
			try:
				# print(nodes[j])
				_, j_1, j_2  = nodes[j].split()
			except:
				break
			
			n_2 = np.array((float(j_1), float(j_2)))
			dist = np.linalg.norm(n_1 - n_2)

			node_dict[i+1][j+1] = dist
			node_dict[i+1] = node_dict[i+1]
			
			node_dict[j+1][i+1] = dist
			node_dict[j+1] = node_dict[j+1]
	
	return node_dict

# Random restart
def greedy_solution(dist_dict, state=[], cost=[]):
	for d in dist_dict:
		dist_dict[d] = dict(sorted(dist_dict[d].items(), key = itemgetter(1)))
	
	if state == []:
		start = randint(1, len(dist_dict))
		state.append(start) 

	if cost == []:
		cost = [0]
	
	while len(state) < len(dist_dict):
		state.append([i for i in dist_dict[state[-1]] if i not in state][0])
		cost.append(cost[-1] + dist_dict[state[-1]][state[-2]])

	return state, cost

# Random 
def random_modification(dist_dict, state, cost):
	i = randint(2, len(state)-1)
	state, nodes = state[:i], state[i:]
	new_cost = cost[:i]
	state.append(sample(nodes, 1)[0])
	new_cost.append(new_cost[-1] + dist_dict[state[-1]][state[-2]])
	return greedy_solution(dist_dict, state=state, cost=new_cost)

# Neighbours
def selected_modification(dist_dict, i, state, cost):
	state, nodes = state[:i], state[i:]
	new_cost = cost[:i]
	state.append(sample(nodes, 1)[0])
	new_cost.append(new_cost[-1] + dist_dict[state[-1]][state[-2]])
	return greedy_solution(dist_dict, state=state, cost=new_cost)

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
	
	lines = lines[6:-1]
	
	dist_dict = to_dict(lines)
	
	state, cost = greedy_solution(dist_dict)
	print(cost[-1])
	for _ in tqdm(range(100)):
		new_state, new_cost = greedy_solution(dist_dict, state=[], cost=[])
		if cost[-1] > new_cost[-1] :
			state, cost = new_state, new_cost
	print(cost[-1],'\n',new_state)
	
	
	
	for _ in tqdm(range(100)):
		new_state, new_cost = random_modification(dist_dict, state, cost)
		if cost[-1] > new_cost[-1] :
			state, cost = new_state, new_cost
	print(cost[-1],'\n',new_state)


	for _ in tqdm(range(100)):
		i = randint(len(state)-len(state)//10, len(state)-1)
		new_state, new_cost = selected_modification(dist_dict, i, state, cost)
		if cost[-1] > new_cost[-1] :
			state, cost = new_state, new_cost
	print(cost[-1],'\n',new_state)


	f = open("solution.csv", "w")
	for x in state:
		f.write(str(x)+'\n')
	f.close()

	cost = 0
	for x in range(len(dist_dict)-1):
		cost+= dist_dict[state[x]][state[x+1]]

	print(cost)
	

if __name__ == "__main__":
	main()
	print("End")
