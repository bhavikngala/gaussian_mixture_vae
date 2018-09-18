import torch
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
import math
import argparse
import os
import sys

from networks.GMVAE_OG import GMVAE
from networks.tvsumNet import *

import dataloader as dl
import utils

parser = argparse.ArgumentParser(description='Gaussian Mixture VAE')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--K', type=int, metavar='N',
					help='number of clusters')
parser.add_argument('--x-size', type=int, default=200, metavar='N',
					help='dimension of x')
parser.add_argument('--hidden-size', type=int, default=512, metavar='N',
					help='dimension of hidden layer')
parser.add_argument('--w-size', type=int, default=150, metavar='N',
					help='dimension of w')
parser.add_argument('--dataset', help='dataset to use')
parser.add_argument('--feature-type', default='c3d', help='dataset to use')
parser.add_argument('--learning-rate', type=float, default=1e-4,
					help='learning rate for optimizer')
parser.add_argument('--continuous', help='data is continuous',
					action='store_true')

args = parser.parse_args()

# select gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.device = device

torch.manual_seed(args.seed)

if args.dataset == 'mnist':
	train_loader, test_loader = dl.mnistloader(args.batch_size)
elif args.dataset == 'tvsum' and args.feature_type == 'gist':
	train_loader, test_loader = dl.tvsumloader(args.batch_size,
		'./data/tvsum.mat', 'gist')
elif args.dataset == 'tvsum':
	train_loader, test_loader = dl.tvsumloader(args.batch_size,
		'./data/tvsum.mat', 'c3d')
elif args.dataset == 'toy':
	train_loader, test_loader = dl.toyloader(args.batch_size,
		args.K, 4096, 40000)
elif args.dataset == 'spiral':
	train_loader, test_loader = dl.spiralloader(args.batch_size,
		'./data/spiral.npy')

if args.dataset == 'mnist':
	gmvae = GMVAE(args).to(device)
elif args.dataset == 'tvsum' and args.feature_type == 'gist':
	gmvae = Tvsum_gist(args).to(device)
elif args.dataset == 'tvsum':
	gmvae = Tvsum(args).to(device)
elif args.dataset == 'toy':
	gmvae = Tvsum(args).to(device)
elif args.dataset == 'spiral':
	gmvae = Spiral(args).to(device)

optimizer = optim.Adam(gmvae.parameters(), lr=args.learning_rate)

lam = torch.Tensor([100.0]).to(device)
def loss_function(recon_X, X, mu_w, logvar_w, qz,
	mu_x, logvar_x, mu_px, logvar_px, x_sample):
	N = X.size(0) # batch size

	# 1. Reconstruction Cost = -E[log(P(y|x))]
	# for dataset such as mnist
	if not args.continuous:
		recon_loss = F.binary_cross_entropy(recon_X, X,
			size_average=False)
	# for datasets such as tvsum, spiral
	elif args.continuous:
		# unpack Y into mu_Y and logvar_Y
		mu_recon_X, logvar_recon_X = recon_X

		# use gaussian criteria
		# negative LL, so sign is flipped
		# log(sigma) + 0.5*2*pi + 0.5*(x-mu)^2/sigma^2
		recon_loss = 0.5 * torch.sum(logvar_recon_X + math.log(2*math.pi) \
			+ (X - mu_recon_X).pow(2)/logvar_recon_X.exp())

	# 2. KL( q(w) || p(w) )
	KLD_W = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())

	# 3. KL( q(z) || p(z) )
	KLD_Z = torch.sum(qz * torch.log(args.K * qz + 1e-10))
	if args.dataset == 'spiral':
		KLD_Z = max(lam, KLD_Z)

	# 4. E_z_w[KL(q(x)|| p(x|z,w))]
	# KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
	mu_x = mu_x.unsqueeze(-1)
	mu_x = mu_x.expand(-1, args.x_size, args.K)

	logvar_x = logvar_x.unsqueeze(-1)
	logvar_x = logvar_x.expand(-1, args.x_size, args.K)

	# shape (-1, x_size, K)
	KLD_QX_PX = 0.5 * (((logvar_px - logvar_x) + \
		((logvar_x.exp() + (mu_x - mu_px).pow(2))/logvar_px.exp())) \
		- 1)

	# transpose to change dim to (-1, x_size, K)
	# KLD_QX_PX = KLD_QX_PX.transpose(1,2)
	qz = qz.unsqueeze(-1)
	qz = qz.expand(-1, args.K, 1)

	E_KLD_QX_PX = torch.sum(torch.bmm(KLD_QX_PX, qz))

	# 5. Entropy criterion
	
	# CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]
	# compute likelihood
	
	x_sample = x_sample.unsqueeze(-1)
	x_sample =  x_sample.expand(-1, args.x_size, args.K)

	temp = 0.5 * args.x_size * math.log(2 * math.pi)
	# log likelihood
	llh = -0.5 * torch.sum(((x_sample - mu_px).pow(2))/logvar_px.exp(), dim=1) \
			- 0.5 * torch.sum(logvar_px, dim=1) - temp

	lh = F.softmax(llh, dim=1)

	# entropy
	CV = torch.sum(torch.mul(torch.log(lh+1e-10), lh))
	
	loss = recon_loss + KLD_W + KLD_Z + E_KLD_QX_PX
	return loss, recon_loss, KLD_W, KLD_Z, E_KLD_QX_PX, CV

def train(epoch):
	gmvae.train()

	statusString = 'Train epoch: {:5d}[{:5d}/{:5d} loss: {:.6f} ReconL: {:.6f} E(KLD(QX||PX)): {:.6f} CV: {:.6f} KLD_W: {:.6f} KLD_Z: {:.6f} accuracy: {:.6f}]\n'
	acc = 0.0
	for batch_idx, (data, target) in enumerate(train_loader):
		data = data.to(device)

		target = target.to(device)
		optimizer.zero_grad()

		mu_x, logvar_x, mu_px, logvar_px, qz, Y , mu_w, logvar_w, \
			x_sample = gmvae(data)

		loss, BCE, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
			= loss_function(Y, data, mu_w, logvar_w, qz,
			mu_x, logvar_x, mu_px, logvar_px, x_sample)

		loss.backward()

		optimizer.step()

		accuracy = 0.0
		if args.dataset == 'mnist' or args.dataset == 'toy':
			x_n = qz.argmax(0)
			labels = qz.argmax(1)


			for k in range(args.K):
				counts = torch.sum(((labels==k)+(target==target[x_n[k]]))==2)
				accuracy = accuracy + counts.item()
		
			acc = acc + accuracy
			accuracy = accuracy / len(data)

		status = statusString.format(epoch, batch_idx+1, len(train_loader),
					loss.item(), BCE.item(), E_KLD_QX_PX.item(),
					CV.item(), KLD_W.item(), KLD_Z.item(), accuracy)
		utils.writeStatusToFile(trainStatusFile, status)

def test(epoch):
	gmvae.eval()

	statusString = 'Test epoch: {:5d}[{:5d}/{:5d} loss: {:.4f} ReconL: {:.4f} E(KLD(QX||PX)): {:.4f} CV: {:.4f} KLD_W: {:.4f} KLD_Z: {:.4f} accuracy: {:.4f}]\n'
	acc = 0.0
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data = data.to(device)
			
			target = target.to(device)

			mu_x, logvar_x, mu_px, logvar_px, qz, Y , mu_w, logvar_w, \
				x_sample = gmvae(data)

			loss, BCE, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
				= loss_function(Y, data, mu_w, logvar_w, qz,
				mu_x, logvar_x, mu_px, logvar_px, x_sample)

			accuracy = 0.0
			if args.dataset == 'mnist' or args.dataset == 'toy':
				x_n = qz.argmax(0)
				labels = qz.argmax(1)


				for k in range(args.K):
					counts = torch.sum(((labels==k)+(target==target[x_n[k]]))==2)
					accuracy = accuracy + counts.item()
			
				acc = acc + accuracy
				accuracy = accuracy / len(data)

			status = statusString.format(epoch, batch_idx+1, len(test_loader),
					loss.item(), BCE.item(), E_KLD_QX_PX.item(), CV.item(),
					KLD_W.item(), KLD_Z.item(), accuracy)
			
			utils.writeStatusToFile(testStatusFile, status)

			if args.dataset == 'mnist' and batch_idx == 0:
				n = min(data.size(0), 32)
				comparision = torch.cat([data[:n], Y[:n]])

				save_image(comparision.cpu(),
					'./results/reconstruction_'+str(epoch)+'.png', nrow=8)

# gmvae.apply(utils.weights_init)

model_file = './model/' + args.dataset + '_' + str(args.K) + '_gmvae.pth'
trainStatusFile = './losses/' + args.dataset + '_' + str(args.K) + '_train.txt'
testStatusFile = './losses/' + args.dataset + '_' + str(args.K) + '_test.txt'

if args.dataset == 'tvsum':
	model_file = './model/' + args.dataset + '_' + args.feature_type + '_' + str(args.K) + '_gmvae.pth'
	trainStatusFile = './losses/' + args.dataset + '_' + args.feature_type + '_' + str(args.K) + '_train.txt'
	testStatusFile = './losses/' + args.dataset + '_' + args.feature_type + '_' + str(args.K) + '_test.txt'

if os.path.isfile(model_file):
	gmvae.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

for epoch in range(1, args.epochs+1):
	# train the network
	train(epoch)
	# test the network
	test(epoch)

	if epoch % 10 == 0:
		torch.save(gmvae.state_dict(), model_file)