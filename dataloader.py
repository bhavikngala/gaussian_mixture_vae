import torch
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
import numpy as np
import pickle


def mnistloader(batchSize):
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
			'./data', train=True, download=True,
			transform=transforms.ToTensor()
		),
		batch_size=batchSize,
		shuffle=True
	)

	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
			'./data', train=False, download=True,
			transform=transforms.ToTensor()
		),
		batch_size=batchSize,
		shuffle=True
	)

	return train_loader, test_loader

class TvsumDataset(Dataset):

	def __init__(self, path, dataType, train=True):

		# read the matlab file and load the required feature vector
		self.data = loadmat(path).get('data')[dataType]

		if train:
			# first 40 videos for training
			numsamples = 40
		else:
			# remaining samples for test
			numsamples = 10

		dtemp = []

		for i in range(numsamples):
			# negative indices for test samples
			if not train:
				i = -i - 1

			d = self.data[0][i]
			for j in range(d.shape[0]):
				dtemp.append(d[j])

		self.data = np.array(dtemp)
		# change the datatype to float32
		self.data = self.data.astype('float32')
		
		# convert to tensor
		self.data = torch.from_numpy(self.data)


	def __getitem__(self, index):
		return (self.data[index], index)

	def __len__(self):
		return self.data.size(0)

def tvsumloader(batchSize, path, dataType):
	train_loader = torch.utils.data.DataLoader(
		TvsumDataset(path, dataType, train=True),
		batch_size=batchSize,
		shuffle=True
	)

	test_loader = torch.utils.data.DataLoader(
		TvsumDataset(path, dataType, train=False),
		batch_size=batchSize,
		shuffle=True
	)

	return train_loader, test_loader

class RandomSampleDataset(Dataset):

	def __init__(self, numClusters, sampleDim, numSamples):
		# number of distributions
		self.numClusters = numClusters

		# dimension of the sample in distribution
		self.sampleDim = sampleDim

		# number of samples
		self.numSamples = numSamples

		# mu and var for those distributions
		self.mu = []
		self.var = []

		temp = [5, 10, 15]
		for i in range(numClusters):
			# torch.manual_seed(i)
			# self.mu.append(torch.randn(sampleDim))
			self.mu.append(torch.ones(sampleDim) * temp[i])
			self.var.append(torch.ones(sampleDim))

		# categorical distribution to sample cluster; clusters have equal probabilities
		self.dist = torch.distributions.Categorical(torch.ones(1, numClusters)/numClusters)

	def __getitem__(self, _):
		# sample the cluster from categorical distribution
		cluster = self.dist.sample()

		# sample from normal distribution
		return torch.normal(self.mu[int(cluster)], torch.sqrt(self.var[int(cluster)])), cluster


	def __len__(self):
		return self.numSamples

def toyloader(batchSize, numClusters, sampleDim, numSamples):
	train_loader = torch.utils.data.DataLoader(
			RandomSampleDataset(numClusters, sampleDim, int(0.8 * numSamples)),
			batch_size = batchSize,
			shuffle = True
		)

	test_loader = torch.utils.data.DataLoader(
			RandomSampleDataset(numClusters, sampleDim, int(0.2 * numSamples)),
			batch_size = batchSize,
			shuffle = True
		)

	return train_loader, test_loader

class SpiralDataset(Dataset):

	def __init__(self, filepath, train=True):
		self.data = torch.from_numpy(np.load(filepath))

		if not train:
			self.data = self.data[int(0.8 * self.data.size(0)):]
		else:
			self.data = self.data[:int(0.8 * self.data.size(0))]

	def __getitem__(self, index):
		return (self.data[index], index)

	def __len__(self):
		return self.data.size(0)

def spiralloader(batchSize, filepath):
	train_loader = torch.utils.data.DataLoader(
			SpiralDataset(filepath, train=True),
			batch_size = batchSize,
			shuffle = True
		)

	test_loader = torch.utils.data.DataLoader(
			SpiralDataset(filepath, train=False),
			batch_size = batchSize,
			shuffle = True
		)

	return train_loader, test_loader

f = open('./data/CIFAR10/cifar-10-batches-py/test_batch', 'rb')
test_files = pickle.load(f, encoding='bytes')
test_filenames = test_files['filenames'.encode('utf-8')]
test_filenames = [s.decode('utf-8').strip() for s in test_filenames]

def createCIDAR10GistDataset():
	# filenames of the test batch
	f = open('./data/CIFAR10/cifar-10-batches-py/test_batch', 'rb')
	test_files = pickle.load(f, encoding='bytes')
	test_filenames = test_files['filenames'.encode('utf-8')]
	test_filenames = [s.decode('utf-8').strip() for s in test_filenames]

	filewritten = 0

	gistFiles = ['gist_car.txt',
				 'gist_dog.txt',
				 'gist_deer.txt',
				 'gist_plane.txt',
				 'gist_cat.txt',
				 'gist_horse.txt',
				 'gist_ship.txt',
				 'gist_truck.txt',
				 'gist_bird.txt',
				 'gist_frog.txt']

	filenames = ['filenames_car.txt',
				 'filenames_dog.txt',
				 'filenames_deer.txt',
				 'filenames_plane.txt',
				 'filenames_cat.txt',
				 'filenames_horse.txt',
				 'filenames_ship.txt',
				 'filenames_truck.txt',
				 'filenames_bird.txt',
				 'filenames_frog.txt']

	for i in range(len(gistFiles)):
		print(gistFiles[i])
		cat = gistFiles[i][5:-4]
		gistFile = open('./data/cifar10/'+gistFiles[i], 'r')
		filenameFile = open('./data/cifar10/'+filenames[i], 'r')

		gist = gistFile.readline()
		filename = filenameFile.readline().split('/')[-1].strip()

		while gist:
			gistValues = gist[1:-1].strip().split(' ')
			tempArr = np.array([float(a.strip()) for a in gistValues])

			# if filename in test_filenames:
			# 	tempName = './data/cifar10/test/' + filename[:-4] + '_' + cat
			# 	np.save(tempName, tempArr)
			# 	filewritten = filewritten + 1
			if filename not in test_filenames:
				tempName = './data/cifar10/train/' + filename[:-4] + '_' + cat
				np.save(tempName, tempArr)
				filewritten = filewritten + 1

			gist = gistFile.readline()
			filename = filenameFile.readline().split('/')[-1].strip()			
	print('filewritten:', filewritten)