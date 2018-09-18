import torch
from torch import nn
from torch.nn import functional as F
from networks import *

class GMVAE(nn.Module):
	'''
	Gaussian Mixture Variational Autoencoder.
	Ref: Deep Unsupervised Clustering with 
	Gaussian Mixture Variational Autoencoders.
	https://arxiv.org/abs/1611.02648
	'''

	def __init__(self, args):
		'''
		params: K; cluster size
				x_size; dimension of latent variable x
				w_size; dimension of latent variable w
				hidden_size; dimension of hidden layer
				dataset; name of the dataset
				device: gpu or cpu
		'''
		super(GMVAE, self).__init__()

		self.K = args.K
		self.x_size = args.x_size
		self.w_size = args.w_size
		self.hidden_size = args.hidden_size
		self.device = args.device
		self.dataset = args.dataset

		if self.dataset == 'mnist':
			# encoder layers
			self.convStack, self.fcStack = \
				mnistConv_Encoder(self.hidden_size)
			# prior generator layers
			self.priorGenStack = \
				mnist_PriorGen(self.w_size, self.hidden_size)
			# decoder layers
			self.decoderConvStack, self.decoderFCStack = \
				mnistConv_Decoder(self.x_size, self.hidden_size)
			
		# output of encoder; [X] -> [x], [w], [z]
		self.fc_mu_x = nn.Linear(self.hidden_size, self.x_size)
		self.fc_logvar_x = nn.Linear(self.hidden_size, self.x_size)
		self.fc_mu_w = nn.Linear(self.hidden_size, self.w_size)
		self.fc_logvar_w = nn.Linear(self.hidden_size, self.w_size)
		self.fc_qz = nn.Linear(self.hidden_size, self.K)

		# prior x for each cluster
		self.fc_mu_px = nn.ModuleList(
			[nn.Linear(self.hidden_size, self.x_size) for i in range(self.K)])
		self.fc_logvar_px = nn.ModuleList(
			[nn.Linear(self.hidden_size, self.x_size) for i in range(self.K)])

	def encoder(self, X):
		'''
		Encoder graph, takes data as input,
		encodes data to latent variables w, x, and z.
		[X] -> [x], [w], [z]
		z is a vector with length equal to K.
		x is a sample from mixture of Gaussians, 
		with the mode corresponding to its class as dominant.
		w is a latent variable from N(0, I) distribution
		'''
		c1 = self.convStack(X)
		h1 = self.fcStack(c1.view(-1, self.hidden_size))
		
		mu_x = self.fc_mu_x(h1)
		logvar_x = self.fc_logvar_x(h1)
		
		mu_w = self.fc_mu_w(h1)
		logvar_w = self.fc_logvar_w(h1)
		
		qz = F.softmax(self.fc_qz(h1))

		return mu_x, logvar_x, mu_w, logvar_w, qz

	def priorGenerator(self, w_sample, batchSize):
		'''
		Generate Gaussians based on latent variable w sample
		[w] -> {[x]}2K
		'''
		h1 = F.tanh(self.priorGenStack(w_sample))

		mu_px = torch.empty(batchSize, self.K, self.x_size,
			device=self.device, requires_grad=False)
		logvar_px = torch.empty(batchSize, self.K, self.x_size,
			device=self.device, requires_grad=False)

		for i in range(self.K):
			mu_px[:, i, :] = self.fc_mu_px[i](h1)
			logvar_px[:, i, :] = self.fc_logvar_px[i](h1)

		return mu_px, logvar_px

	def decoder(self, x_sample):
		'''
		Decoder graph, reconstructs X from latent variable x
		[x] -> [Y]
		'''
		h1 = self.decoderFCStack(x_sample)

		# TODO something about the view function below
		h1 = h1.view(-1, 8, 8, 8)
		Y = self.decoderConvStack(h1)
		return Y

	def reparameterize(self, mu, logvar):
		'''
		compute z = mu + std * epsilon
		'''
		if self.training:
			# do this only while training
			# compute the standard deviation from logvar
			std = torch.exp(0.5 * logvar)
			# sample epsilon from a normal distribution with mean 0 and
			# variance 1
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, X):
		batchSize = X.size(0)

		mu_x, logvar_x, mu_w, logvar_w, qz = self.encoder(X)
		w_sample = self.reparameterize(mu_w, logvar_w)
		mu_px, logvar_px = self.priorGenerator(
			w_sample, batchSize)

		x_sample = self.reparameterize(mu_x, logvar_x)
		Y = self.decoder(x_sample)

		return mu_x, logvar_x, mu_px, logvar_px, qz, Y, mu_w, \
			logvar_w, x_sample