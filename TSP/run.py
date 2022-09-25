# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114

import argparse
from typing import Dict
import numpy as np
import re
from random import randint, sample
from operator import itemgetter
from tqdm import tqdm

def to_dict(nodes):
	
	node_dict = {n+1:{_n+1:{} for _n in range(len(nodes))} for n in range(len(nodes))}
	
	for i in range(len(nodes)):
		_, i_1, i_2 = nodes[i].split()
		n_1 = np.array((int(i_1), int(i_2)))
		
		for j in range(i, len(nodes)):
			_, j_1, j_2  = nodes[j].split()
			
			n_2 = np.array((int(j_1), int(j_2)))
			dist = np.linalg.norm(n_1 - n_2)

			node_dict[i+1][j+1] = dist
			node_dict[i+1] = node_dict[i+1]
			
			node_dict[j+1][i+1] = dist
			node_dict[j+1] = node_dict[j+1]
	
	return node_dict

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
	
def selected_modification(dist_dict, i, state, cost):
	state, nodes = state[:i], state[i:]
	new_cost = cost[:i+1]
	state.append(sample(nodes, 1)[0])
	return greedy_solution(dist_dict, state=state, cost=new_cost)

def random_modification(dist_dict, state, cost):
	i = randint(0, len(state)-1)
	state, nodes = state[:i], state[i:]
	new_cost = cost[:i+1]
	state.append(sample(nodes, 1)[0])
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
	for _ in tqdm(range(10)):
		new_state, new_cost = greedy_solution(dist_dict, state=[], cost=[])
		if cost[-1] > new_cost[-1] :
			state, cost = new_state, new_cost

	print(cost[-1],'\n',new_state)
	
	
	
	for _ in tqdm(range(10)):
		new_state, new_cost = random_modification(dist_dict, state, cost)
		if cost[-1] > new_cost[-1] :
			state, cost = new_state, new_cost
	print(cost[-1],'\n',new_state)
	

		
	for _ in tqdm(range(10)):
		i = randint(len(state)-len(state)//10, len(state)-1)
		new_state, new_cost = selected_modification(dist_dict, i, state, cost)
		if cost[-1] > new_cost[-1] :
			state, cost = new_state, new_cost
	print(cost[-1],'\n',new_state)


	f = open("solution.csv", "w")
	for x in state:
		f.write(str(x)+'\n')
	f.close()


	sw = state

	s = [1, 2, 242, 243, 244, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 246, 245, 247, 250, 251, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 207, 206, 205, 204, 203, 202, 201, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 176, 180, 179, 150, 178, 177, 151, 152, 156, 153, 155, 154, 129, 130, 131, 20, 21, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 157, 158, 159, 160, 175, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 171, 173, 174, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 109, 108, 110, 111, 112, 88, 87, 113, 114, 115, 117, 116, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 59, 63, 62, 118, 61, 60, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 22, 25, 23, 24, 14, 15, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 277, 276, 275, 274, 273, 272, 271, 16, 17, 18, 19, 132, 133, 134, 270, 269, 135, 136, 268, 267, 137, 138, 139, 149, 148, 147, 146, 145, 199, 200, 144, 143, 142, 141, 140, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257, 254, 253, 208, 209, 252, 255, 256, 249, 248, 278, 279, 3, 280]
	cost = 0
	for x in range(len(s)-1):
		cost+= dist_dict[s[x]][s[x+1]]

	print(cost)

	cost = 0
	for x in range(len(sw)-1):
		cost+= dist_dict[sw[x]][sw[x+1]]

	print(cost)




















































if __name__ == "__main__":
	main()
	print("End")
	
