import numpy as np

class Zeros:
	def __init__(self):
		pass

	def create_weights(self,size):
		return np.zeros(size)

class Ones:
	def __init__(self):
		pass

	def create_weights(self,size):
		return np.ones(size)


class Uniform:
	def __init__(self,low=-1,high=1):
		self.low=low
		self.high=high

	def create_weights(self,size):
		return np.random.uniform(low=self.low,high=self.high,size=size)



init_dict={
	'zeros':Zeros,
	'ones':Ones,
	'uniform':Uniform
}