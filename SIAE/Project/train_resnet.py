# Author	: Cristopher McIntyre Garcia 
# Email		: cmcin019@uottawa.ca
# S-N		: 300025114

# Run:
# python3 run.py -l -t 4 -s -g


# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from tqdm import tqdm 
from os import system
import os

# Import models from timm
# from resnet import resnet50
# from vision_transformer import vit_small_patch16_224
from timm.models.resnet import resnet50 as cnn
from timm.models.resnet import resnetrs101 as cnn_101
from timm.models.vision_transformer import vit_base_patch8_224 as vit

# Torch imports 
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import torchvision

torch.cuda.is_available()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Training Hyperparameters
learning_rate = 0.001
num_epochs = 5

# PGD Hyperparameters
epsilon=0.3
steps=5
alpha=0.02


def load_data(d):
	if d == 'm':
		# Load training dataset
		train_dataset = datasets.MNIST(
			root = 'data',
			train = True,
			transform = transforms.ToTensor(),
			download = True,
		)
		train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)

		# Load testing dataset
		test_dataset = datasets.MNIST(
			root = 'data',
			train = False,
			transform = transforms.ToTensor(),
			download = False,
		)
		test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, drop_last=True)

		# Load perturbing training dataset
		pert_dataset = datasets.MNIST(
			root = 'data',
			train = True,
			transform = transforms.ToTensor(),
			download = True,
		)
		return train_loader, test_loader, pert_dataset, 'MNIST'
	else:
		# Load training dataset
		train_dataset = datasets.CIFAR10(
			root = 'data',
			train = True,
			transform = transforms.ToTensor(),
			download = True,
		)
		train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)

		# Load testing dataset
		test_dataset = datasets.CIFAR10(
			root = 'data',
			train = False,
			transform = transforms.ToTensor(),
			download = False,
		)
		test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, drop_last=True)

		# Load perturbing training dataset
		pert_dataset = datasets.CIFAR10(
			root = 'data',
			train = True,
			transform = transforms.ToTensor(),
			download = True,
		)
		return train_loader, test_loader, pert_dataset, 'CIFAR10'

# Load training dataset
train_dataset = datasets.MNIST(
	root = 'data',
	train = True,
	transform = transforms.ToTensor(),
	download = True,
)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)

# Load testing dataset
test_dataset = datasets.MNIST(
	root = 'data',
	train = False,
	transform = transforms.ToTensor(),
	download = False,
)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, drop_last=True)

# Load perturbing training dataset
pert_dataset = datasets.MNIST(
	root = 'data',
	train = True,
	transform = transforms.ToTensor(),
	download = True,
)


def train(model):
	model.to(device=device)
	acc_list = []
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	accuracy = 0
	for epoch in range(num_epochs):
		system('cls' if os.name == 'nt' else 'clear')
		print(f'Training {model.__class__.__name__}')
		print(f'Epoch {epoch}: {accuracy}')
		for _, (data, targets) in enumerate(tqdm(train_loader)):
			data = data.to(device=device)
			targets = targets.to(device=device)

			scores = model(data)
			loss = criterion(scores, targets)

			optimizer.zero_grad()
			loss.backward()

			optimizer.step()

		accuracy = model_accuracy(model)
		acc_list.append(accuracy)
	
	print(f'Final accuracy: {accuracy}')
	if device=='cuda:0':
		model.to(device='cpu')
	return acc_list

def model_accuracy(model, pgd=None, pgd_per=[0,0,0], plot=False, target_acc=False, on_training=False):
	real_correct = 0
	correct = 0

	real_oracle_correct = 0
	oracle_correct = 0

	real_brother_correct = 0
	brother_correct = 0
	
	total = 0
	loader = train_loader if on_training else test_loader
	model.eval()
	it = 0
	for images, labels in tqdm(loader):
		it +=1
		images = images.to(device=device)
		images_saved = images
		labels = labels.to(device=device)
		if not pgd == None:
			images, acc_list = pgd(images, labels, model, *pgd_per)
		
		with torch.no_grad():
			real_outputs = model(images_saved)
			_, real_predicted = torch.max(real_outputs.data, 1)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)

			real_oracle_outputs = oracle(images_saved)
			_, real_oracle_predicted = torch.max(real_oracle_outputs.data, 1)			
			oracle_outputs = oracle(images)
			_, oracle_predicted = torch.max(oracle_outputs.data, 1)

			real_brother_outputs = brother(images_saved)
			_, real_brother_predicted = torch.max(real_brother_outputs.data, 1)			
			brother_outputs = brother(images)
			_, brother_predicted = torch.max(brother_outputs.data, 1)

			if plot:
				np_images_s = torchvision.utils.make_grid(images_saved.cpu().data, normalize=True).numpy()
				fig_s, ax_s = plt.subplots()
				ax_s.imshow(np.transpose(np_images_s,(1,2,0)))
				fig_s.savefig(f'images/resnet/{pgd.__name__}/fig_{it}_original.jpg', bbox_inches='tight', dpi=150)
				
				np_images = torchvision.utils.make_grid(images.cpu().data, normalize=True).numpy()
				fig, ax = plt.subplots()
				ax.imshow(np.transpose(np_images,(1,2,0)))
				fig.savefig(f'images/resnet/{pgd.__name__}/fig_{it}_attack.jpg', bbox_inches='tight', dpi=150)

				fig_acc, ax_acc = plt.subplots()
				ax_acc.plot([n for n in range(len(acc_list))], acc_list, label='Accuracy')
				ax_acc.legend()
				title = pgd.__name__.replace('_', ' ').title()
				ax_acc.set_title(f'{title} Accuracy')
				ax_acc.set(xlabel='Iteration', ylabel='Accuracy')
				fig_acc.savefig(f'images/resnet/{pgd.__name__}/Accuracy_{it}.jpg', bbox_inches='tight', dpi=150)
								
				# plot = False

			total += labels.size(0)
			if device=='cuda:0':
				real_correct += (real_predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()
				correct += (predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()

				real_oracle_correct += (real_oracle_predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()
				oracle_correct += (oracle_predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()

				real_brother_correct += (real_brother_predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()			
				brother_correct += (brother_predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()
			else:
				real_correct += (real_predicted==labels).sum().item()
				correct += (predicted==labels).sum().item()

				real_oracle_correct += (real_oracle_predicted==labels).sum().item()
				oracle_correct += (oracle_predicted==labels).sum().item()

				real_brother_correct += (real_brother_predicted==labels).sum().item()
				brother_correct += (brother_predicted==labels).sum().item()
			
			real_TestAccuracy = 100 * real_correct / total
			TestAccuracy = 100 * correct / total

			real_oracle_TestAccuracy = 100 * real_oracle_correct / total
			oracle_TestAccuracy = 100 * oracle_correct / total

			real_brother_TestAccuracy = 100 * real_brother_correct / total
			brother_TestAccuracy = 100 * brother_correct / total
			if pgd != None:

				fp = open(f'images/resnet/{pgd.__name__}/fig_{it}_log.txt', 'w')
				fp.write(f'Model_R:    \t{real_TestAccuracy}\nModel_A:    \t{TestAccuracy}\nOracle_R:  \t{real_oracle_TestAccuracy}\nOracle_A:  \t{oracle_TestAccuracy}\nBrother_R: \t{real_brother_TestAccuracy}\nBrother_A: \t{brother_TestAccuracy}\n\n')
				fp.close()

				print(f'Model_R:    \t{real_TestAccuracy}')
				print(f'Model_A:    \t{TestAccuracy}')

				print(f'Oracle_R:  \t{real_oracle_TestAccuracy}')
				print(f'Oracle_A:  \t{oracle_TestAccuracy}')

				print(f'Brother_R: \t{real_brother_TestAccuracy}')
				print(f'Brother_A: \t{brother_TestAccuracy}')
				print()
				# print(labels)

	model.train()
	# print(real_TestAccuracy)
	# print(TestAccuracy)
	# print(oracle_TestAccuracy)
	# print(brother_TestAccuracy)
	# print(labels)
	return(TestAccuracy)

def boundary_attack(images, labels, model, epsilon=epsilon, steps=steps, alpha=alpha, start=None):
	b, c, height, width = images.shape
	E = 2
	if start != None:
		x, y = start
	else:
		while(True):
			boo = False
			pert_loader = DataLoader(dataset=pert_dataset, batch_size=64, shuffle=True, drop_last=True)
			for _, (x, y) in enumerate(pert_loader):
				x, y = x.cuda(), y.cuda()
				if not True in (labels == y):
					boo = True
					break
			if boo:
				break
	# print(labels)
	# print(y)
	# print()
	y_prev = y
	acc_list = []
	for _ in range(steps):
		# print('start')
		i = 0 
		E /= 2
		while (True):
			i +=1
			compare = y == labels
			if not False in compare:
				y = y_prev
				break
			y_prev = y
			if i == 20:
				break
			compare = 1 - compare.to(torch.float32)
			a = torch.rand_like(images)*E
			compare = torch.transpose(compare.expand(height,c,width,b), 0, 3)
			a = torch.mul(a, compare)
			tmp = torch.mul(images,a) + torch.mul(x, 1 - a)
			out = model(tmp)
			_, y = torch.max(out.data, 1)

			correct = (y==labels).sum().item()
			accuracy = 100 * correct / labels.size(0)
			# print(accuracy)
			if i % 5 == 0:
				acc_list.append(accuracy)

			compare = y == labels
			compare = torch.transpose(compare.expand(height,c,width,b), 0, 3)
			compare = compare.to(torch.float32)
			inv = 1 - compare.to(torch.float32)

			x = torch.mul(compare,x) + torch.mul(inv, tmp)

		i = 0
		while(True):
			i +=1
			compare = y == labels
			if False in compare:
				y_prev = y
				break

			y = y_prev
			if i == 20:
				break

			sig = torch.rand_like(images)*E
			z = x + sig

			compare = compare.to(torch.float32)
			compare = torch.transpose(compare.expand(height,c,width,b), 0, 3)
			eps = torch.abs(images - x)
			eps = torch.mul(eps, compare)

			tmp = torch.min(torch.max(z, images - eps), images + eps).detach()
			out = model(tmp)
			_, y = torch.max(out.data, 1)

			correct = (y==labels).sum().item()
			accuracy = 100 * correct / labels.size(0)
			# print(accuracy)
			if i % 5 == 0:
				acc_list.append(accuracy)

			compare = y == labels
			compare = torch.transpose(compare.expand(height,c,width,b), 0, 3)
			compare = compare.to(torch.float32)
			inv = 1 - compare.to(torch.float32)

			x = torch.mul(compare,x) + torch.mul(inv, tmp)

	# print(labels)
	# print(y)
	return x, acc_list


def inv_boundary_attack(images, labels, model, epsilon=epsilon, steps=steps, alpha=alpha, use_oracle=False):
	b, c, height, width = images.shape
	E = 0.01
	
	x = images
	y = labels
	y_prev = y
	acc_list = []
	for _ in range(steps):

		i = 0
		E += 0.002
		while(True):
			i +=1
			compare = y == labels # Compare prediction with real
			if not True in compare: # If all different
				y_prev = y
				break

			y = y_prev
			if i == 10:
				break

			sig = torch.rand_like(images)
			
			z = x + sig # Small perturbation

			compare_f = compare.to(torch.float32)
			compare_f = torch.transpose(compare_f.expand(height,c,width,b), 0, 3)
			# eps = images + E
			eps = torch.mul(E, compare_f)

			tmp = torch.max(torch.min(z, images - eps), images + eps).detach()
			out = model(tmp)

			_, y = torch.max(out.data, 1)

			correct = (y==labels).sum().item()
			accuracy = 100 * correct / labels.size(0)
			# print(accuracy)
			if i % 5 == 0:
				acc_list.append(accuracy)

			compare = ((y == labels) == False) != (compare == False)

			if use_oracle:
				anchor = oracle(tmp)			
				_, y_anchor = torch.max(anchor.data, 1)
				# oracle_shift = ((y_anchor == labels) == False) * .5
				compare = torch.mul(compare, (y_anchor == labels))

			compare = torch.transpose(compare.expand(height,c,width,b), 0, 3)
			compare = compare.to(torch.float32)
			inv = 1 - compare.to(torch.float32)

			x = torch.mul(inv,x) + torch.mul(compare, tmp)

	print(y)
	return x, acc_list

def combined_boundary_attack(images, labels, model, epsilon=epsilon, steps=steps, alpha=alpha, use_oracle=False):
	x, acc = inv_boundary_attack(images, labels, model, epsilon=epsilon, steps=steps, alpha=alpha, use_oracle=False)
	print('inv')
	_, y = torch.max(model(x).data, 1)
	start = (x,y)
	x, acc_2 = boundary_attack(images, labels, model, epsilon=epsilon, steps=steps, alpha=alpha, start=start)
	acc += acc_2
	return x, acc

def experiment(original_model, model, pgd, plot=False, pgd_per=[]):
	acc_before = model_accuracy(original_model, pgd=pgd, pgd_per=pgd_per, plot=plot)
	print(acc_before)
	acc_attack = None
	# if pgd.__name__ == 'PGD_Targeted':
	# 	acc_attack = model_accuracy(model, pgd=pgd, pgd_per=pgd_per, plot=plot, target_acc=True)
	# acc_list_1, acc_list_2 = train_with_PGD(model, pgd=pgd, pgd_per=pgd_per)
	return (acc_before, [], [], [])

def run(model, plot, spgd, attacks=[], epsilons=[0, .1, .2, .3, .45], steps=1, alpha=0.01):
	original_model = cnn(in_chans=in_chans,num_classes=10).to(device=device)
	original_model.load_state_dict(torch.load('models/' + dataset + '/resnet.pt'))
	acc = []
	for alg in attacks:
		for eps in epsilons:
			if spgd:
				if device=='cuda:0':
					model.to(device='cpu')
				model = cnn(in_chans=in_chans,num_classes=10).to(device=device)
			else:
				model.load_state_dict(torch.load('models/' + dataset + '/resnet.pt'))
			pgd_per = [eps,steps,alpha]
			acc.append((alg.__name__, [eps,steps,alpha], *experiment(original_model, model, alg, plot, pgd_per)))

	return acc, model

def main() -> None :

	global show_graph
	load, plot, spgd, show_graph = args.l, args.p, args.s, args.g

	model = cnn(in_chans=in_chans, num_classes=10)
	if not load:
		acc_list = train(model)
		system('cls' if os.name == 'nt' else 'clear')
		print(model.__class__.__name__)
		for acc in range(len(acc_list)):
			if acc % 2 == 0:
				print(f'Epoch {acc+1}: \t{str(acc_list[acc])}')
			
			torch.save(model.state_dict(), 'models/' + dataset + '/resnet.pt')
	else:
		model = cnn(in_chans=in_chans, num_classes=10)
		model.load_state_dict(torch.load('models/' + dataset + '/resnet.pt'))
		model.to(device=device)

	acc_original = model_accuracy(model)
	runs = []
	models = []

	run1, m = run(model, plot=plot, spgd=spgd, attacks=[combined_boundary_attack], epsilons=[.3], steps=200, alpha=.02)
	
	return 0

if __name__ == "__main__":
	# Arguments and values 
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--task", help="Enter task (1, 2, 3, 4)", type=int, default=-1)
	parser.add_argument("-l", help="Load model", action="store_true", default=False)
	parser.add_argument("-p", help="Plot images", action="store_true", default=False)
	parser.add_argument("-s", help="Train with PGD from scratch", action="store_true", default=False)
	parser.add_argument("-g", help="Show graphed data", action="store_true", default=False)
	parser.add_argument("-d", default='m', type=str, help="Dataset (c or m)")
	args = parser.parse_args()

	train_loader, test_loader, pert_dataset, dataset = load_data(args.d)

	if dataset == 'MNIST':
		img_size=28
		in_chans=1
	else:
		img_size=32
		in_chans=3

	oracle = vit(img_size=img_size,in_chans=in_chans, num_classes=10)
	oracle.load_state_dict(torch.load('models/' + dataset + '/vit.pt'))
	oracle.to(device=device)
	oracle.eval()

	brother = cnn_101(in_chans=in_chans, num_classes=10)
	brother.load_state_dict(torch.load('models/' + dataset + '/resnet_II.pt'))
	brother.to(device=device)
	brother.eval()


	main()
	print("End")
