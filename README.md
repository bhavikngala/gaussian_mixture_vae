# Gaussian Mixture Variational Autoencoders
This project aims at unsupervised clustering through generative models. Thus a variational autoencoder is trained to cluster data in its encoder. Instead of an isotropic gaussian prior, the input is considered to be composed of a mixture of K gaussians, K being the number of clusters the data may posses.

This project is based on the paper:</br>
Nat Dilokthanakul, Pedro A.M. Mediano, Marta Garnelo, Matthew C.H. Lee, Hugh Salimbeni, Kai Arulkumaran, Murray Shanahan. <b>Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders.</b>
</br>Link: https://arxiv.org/abs/1611.02648

This network on MNIST dataset achieves approximately 75% classification accuracy on test set after learning the clusters in training data.
