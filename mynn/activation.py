import numpy as np

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

def linear(x):
	return x

def relu(x):
	return np.maximum(x, 0)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def elu(x,alpha=1):
	x[x<0]=alpha*np.expm1(x[x<0])
	return x

def tanh(x):
	return np.tanh(x)

def sin(x):
	return np.sin(x)

activation_dict={
	'softmax':softmax,
	'linear':linear,
	'relu':relu,
	'elu':elu,
	'tanh':tanh,
	'sigmoid':sigmoid,
	'sin':sin
}