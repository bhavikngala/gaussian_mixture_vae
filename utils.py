# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv')!=-1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm')!=-1:
		m.weight.data.normal_(1.0, 0.02)
		# m.bias.data.fill_(0)


# write strings to file
def writeStatusToFile(filepath, status):
	with open(filepath, 'a') as file:
		file.write(status)