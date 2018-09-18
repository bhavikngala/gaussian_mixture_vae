import torch
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):

	def __init__(self, args):
		super(GMVAE, self).__init__()

		self.device = args.device

		# Recognition model
		self.c1 = nn.Conv2d(1, 16, 6, 1, 0, bias=False)
		self.b1 = nn.BatchNorm2d(16)
		self.c2 = nn.Conv2d(16, 32, 6, 1, 0, bias=False)
		self.b2 = nn.BatchNorm2d(32)
		self.c3 = nn.Conv2d(32, 64, 5, 2, 1, bias=False)
		self.b3 = nn.BatchNorm2d(64)
		self.h1 = nn.Linear(4096, 500, bias=False)
		self.b4 = nn.BatchNorm1d(500)
		self.mu_x = nn.Linear(500, 200)
		self.logvar_x = nn.Linear(500, 200)
		self.mu_w = nn.Linear(500, 150)
		self.logvar_w = nn.Linear(500, 150)
		self.qz = nn.Linear(500, 10)
		
		# prior generator
		self.h2 = nn.Linear(150, 500) # tanh activation
		# prior x for each cluster
		self.mu_px = nn.ModuleList(
			[nn.Linear(500, 200) for i in range(10)])
		self.logvar_px = nn.ModuleList(
			[nn.Linear(500, 200) for i in range(10)])

		# generative model
		self.h3 = nn.Linear(200, 500, bias=False)
		self.b5 = nn.BatchNorm1d(500)
		self.h4 = nn.Linear(200, 4096, bias=False)
		self.b6 = nn.BatchNorm1d(4096)
		self.d3 = nn.ConvTranspose2d(64, 32, 4, 2, 0, bias=False)
		self.b7 = nn.BatchNorm2d(32)
		self.d2 = nn.ConvTranspose2d(32, 16, 6, 1, 0, bias=False)
		self.b8 = nn.BatchNorm2d(16)
		self.d1 = nn.ConvTranspose2d(16, 1, 6, 1, 0)

	def encode(self, X):
		x = self.c1(X)
		x = F.relu(self.b1(x))
		x = self.c2(x)
		x = F.relu(self.b2(x))
		x = self.c3(x)
		x = F.relu(self.b3(x))
		x = x.view(-1, 4096)
		x = self.h1(x)
		x = F.relu(self.b4(x))

		qz = F.softmax(self.qz(x), dim=1)
		mu_x = self.mu_x(x)
		logvar_x = self.logvar_x(x)
		mu_w = self.mu_w(x)
		logvar_w = self.logvar_w(x)

		return qz, mu_x, logvar_x, mu_w, logvar_w

	def priorGenerator(self, w_sample):
		batchSize = w_sample.size(0)

		h = F.tanh(self.h2(w_sample))

		mu_px = torch.empty(batchSize, 200, 10,
			device=self.device, requires_grad=False)
		logvar_px = torch.empty(batchSize, 200, 10,
			device=self.device, requires_grad=False)

		for i in range(10):
			mu_px[:, :, i] = self.mu_px[i](h)
			logvar_px[:, :, i] = self.logvar_px[i](h)

		return mu_px, logvar_px

	def decoder(self, x_sample):
		h = self.h3(x_sample)
		h = F.relu(self.b5(h))
		h = self.h4(x_sample)
		h = F.relu(self.b6(h))

		h = h.view(-1, 64, 8, 8)

		h = self.d3(h)
		h = F.relu(self.b7(h))
		h = self.d2(h)
		h = F.relu(self.b8(h))
		h = self.d1(h)

		Y = F.sigmoid(h)

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
		qz, mu_x, logvar_x, mu_w, logvar_w = self.encode(X)

		w_sample = self.reparameterize(mu_w, logvar_w)
		x_sample = self.reparameterize(mu_x, logvar_x)

		mu_px, logvar_px = self.priorGenerator(w_sample)
		Y = self.decoder(x_sample)

		return mu_x, logvar_x, mu_px, logvar_px, qz, Y, mu_w, \
			logvar_w, x_sample