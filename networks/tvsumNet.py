import torch
from torch import nn
from torch.nn import functional as F

class Tvsum(nn.Module):

	def __init__(self, args):
		super(Tvsum, self).__init__()

		self.args = args

		# Reconstruction model
		self.encode = nn.Sequential(
			nn.Linear(4096, 2048, bias=False),
			nn.BatchNorm1d(2048),
			nn.ReLU(),
			nn.Linear(2048, 1024, bias=False),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 512, bias=False),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512, 512, bias=False),
			nn.BatchNorm1d(512),
			nn.ReLU()
		)
		self.mu_x = nn.Linear(512, self.args.x_size, bias=False)
		self.logvar_x = nn.Linear(512, self.args.x_size, bias=False)
		self.mu_w = nn.Linear(512, self.args.w_size, bias=False)
		self.logvar_w = nn.Linear(512, self.args.w_size, bias=False)
		self.qz = nn.Linear(512, self.args.K, bias=False)

		# prior generator
		self.h1 = nn.Linear(self.args.w_size, 512, bias=False)
		self.mu_px = nn.ModuleList(
			[nn.Linear(512, self.args.x_size, bias=False) for i in range(self.args.K)])
		self.logvar_px = nn.ModuleList(
			[nn.Linear(512, self.args.x_size, bias=False) for i in range(self.args.K)])

		# generative model
		self.decode = nn.Sequential(
			nn.Linear(self.args.x_size, 512, bias=False),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512, 512, bias=False),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512, 1024, bias=False),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 2048, bias=False),
			nn.BatchNorm1d(2048),
			nn.ReLU()
			# nn.Linear(2048, 4096, bias=False)
		)
		self.mu_y = nn.Linear(2048, 4096, bias=False)
		self.logvar_y = nn.Linear(2048, 4096, bias=False)

	def encoder(self, X):
		h = self.encode(X)
		qz = F.softmax(self.qz(h), dim=1)
		mu_x = self.mu_x(h)
		logvar_x = self.logvar_x(h)
		mu_w = self.mu_w(h)
		logvar_w = self.logvar_w(h)

		return qz, mu_x, logvar_x, mu_w, logvar_w

	def priorGenerator(self, w_sample):
		batchSize = w_sample.size(0)

		h = F.tanh(self.h1(w_sample))

		mu_px = torch.empty(batchSize, self.args.x_size, self.args.K,
			device=self.args.device, requires_grad=False)
		logvar_px = torch.empty(batchSize, self.args.x_size, self.args.K,
			device=self.args.device, requires_grad=False)

		for i in range(self.args.K):
			mu_px[:, :, i] = self.mu_px[i](h)
			logvar_px[:, :, i] = self.logvar_px[i](h)

		return mu_px, logvar_px

	def decoder(self, x_sample):
		h = self.decode(x_sample)
		return (self.mu_y(h), self.logvar_y(h))

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
		qz, mu_x, logvar_x, mu_w, logvar_w = self.encoder(X)

		w_sample = self.reparameterize(mu_w, logvar_w)
		x_sample = self.reparameterize(mu_x, logvar_x)

		mu_px, logvar_px = self.priorGenerator(w_sample)
		Y = self.decoder(x_sample)

		return mu_x, logvar_x, mu_px, logvar_px, qz, Y, mu_w, \
			logvar_w, x_sample

class Tvsum_gist(nn.Module):

	def __init__(self, args):
		super(Tvsum_gist, self).__init__()

		self.args = args

		# Reconstruction model
		self.encode = nn.Sequential(
			nn.Linear(512, 256),
			nn.ReLU()
		)
		self.mu_x = nn.Linear(256, self.args.x_size)
		self.logvar_x = nn.Linear(256, self.args.x_size)
		self.mu_w = nn.Linear(256, self.args.w_size)
		self.logvar_w = nn.Linear(256, self.args.w_size)
		self.qz = nn.Linear(256, self.args.K)

		# prior generator
		self.h1 = nn.Linear(self.args.w_size, int(self.args.w_size/2))
		self.mu_px = nn.ModuleList(
			[nn.Linear(int(self.args.w_size/2), self.args.x_size) for i in range(self.args.K)])
		self.logvar_px = nn.ModuleList(
			[nn.Linear(int(self.args.w_size/2), self.args.x_size) for i in range(self.args.K)])

		# generative model
		self.decode = nn.Sequential(
			nn.Linear(self.args.x_size, 256),
			nn.ReLU()
		)
		self.mu_y = nn.Linear(256, 512)
		self.logvar_y = nn.Linear(256, 512)

	def encoder(self, X):
		h = self.encode(X)
		qz = F.softmax(self.qz(h), dim=1)
		mu_x = self.mu_x(h)
		logvar_x = self.logvar_x(h)
		mu_w = self.mu_w(h)
		logvar_w = self.logvar_w(h)

		return qz, mu_x, logvar_x, mu_w, logvar_w

	def priorGenerator(self, w_sample):
		batchSize = w_sample.size(0)

		h = F.tanh(self.h1(w_sample))

		mu_px = torch.empty(batchSize, self.args.x_size, self.args.K,
			device=self.args.device, requires_grad=False)
		logvar_px = torch.empty(batchSize, self.args.x_size, self.args.K,
			device=self.args.device, requires_grad=False)

		for i in range(self.args.K):
			mu_px[:, :, i] = self.mu_px[i](h)
			logvar_px[:, :, i] = self.logvar_px[i](h)

		return mu_px, logvar_px

	def decoder(self, x_sample):
		h = self.decode(x_sample)
		return (self.mu_y(h), self.logvar_y(h))

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
		qz, mu_x, logvar_x, mu_w, logvar_w = self.encoder(X)

		w_sample = self.reparameterize(mu_w, logvar_w)
		x_sample = self.reparameterize(mu_x, logvar_x)

		mu_px, logvar_px = self.priorGenerator(w_sample)
		Y = self.decoder(x_sample)

		return mu_x, logvar_x, mu_px, logvar_px, qz, Y, mu_w, \
			logvar_w, x_sample

class Spiral(nn.Module):

	def __init__(self, args):
		super(Spiral, self).__init__()

		self.args = args

		# Reconstruction model
		self.encode = nn.Sequential(
			nn.Linear(self.args.x_size, 100, bias=False),
			nn.BatchNorm1d(100),
			nn.ReLU(),
			nn.Linear(100, 100, bias=False),
			nn.BatchNorm1d(100),
			nn.ReLU()
		)
		self.mu_x = nn.Linear(100, self.args.x_size, bias=False)
		self.logvar_x = nn.Linear(100, self.args.x_size, bias=False)
		self.mu_w = nn.Linear(100, self.args.w_size, bias=False)
		self.logvar_w = nn.Linear(100, self.args.w_size, bias=False)
		self.qz = nn.Linear(100, self.args.K, bias=False)

		# prior generator
		self.h1 = nn.Linear(self.args.w_size, 100, bias=False)
		self.mu_px = nn.ModuleList(
			[nn.Linear(100, self.args.x_size, bias=False) for i in range(self.args.K)])
		self.logvar_px = nn.ModuleList(
			[nn.Linear(100, self.args.x_size, bias=False) for i in range(self.args.K)])

		# generative model
		self.decode = nn.Sequential(
			nn.Linear(self.args.x_size, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU()
		)
		self.mu_y = nn.Linear(100, args.x_size, bias=False)
		self.logvar_y = nn.Linear(100, args.x_size, bias=False)

	def encoder(self, X):
		h = self.encode(X)
		qz = F.softmax(self.qz(h), dim=1)
		mu_x = self.mu_x(h)
		logvar_x = self.logvar_x(h)
		mu_w = self.mu_w(h)
		logvar_w = self.logvar_w(h)

		return qz, mu_x, logvar_x, mu_w, logvar_w

	def priorGenerator(self, w_sample):
		batchSize = w_sample.size(0)

		h = F.tanh(self.h1(w_sample))

		mu_px = torch.empty(batchSize, self.args.x_size, self.args.K,
			device=self.args.device, requires_grad=False)
		logvar_px = torch.empty(batchSize, self.args.x_size, self.args.K,
			device=self.args.device, requires_grad=False)

		for i in range(self.args.K):
			mu_px[:, :, i] = self.mu_px[i](h)
			logvar_px[:, :, i] = self.logvar_px[i](h)

		return mu_px, logvar_px

	def decoder(self, x_sample):
		h = self.decode(x_sample)
		return (self.mu_y(h), self.logvar_y(h))

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
		qz, mu_x, logvar_x, mu_w, logvar_w = self.encoder(X)

		w_sample = self.reparameterize(mu_w, logvar_w)
		x_sample = self.reparameterize(mu_x, logvar_x)

		mu_px, logvar_px = self.priorGenerator(w_sample)
		Y = self.decoder(x_sample)

		return mu_x, logvar_x, mu_px, logvar_px, qz, Y, mu_w, \
			logvar_w, x_sample