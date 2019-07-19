import numpy as np
from .initializers import init_dict as init_dict_classes
from .activation import activation_dict
class layer:
	def __init__(self):
		self.next_layers=[]
		self.prev_layers=[]
		self.res=None

	def __call__(self,prev_layer):
		self.define_prev_layer(prev_layer)
		self.define_this_layer(prev_layer)
		self.compute_output_shape()
		self.init_weights()
		return self

	def clear_res(self):
		self.res=None

	def define_prev_layer(self,prev_layer):
		if type(prev_layer)==list:
			for prev_l in prev_layer:
				self.prev_layers.append(prev_l)
		else:
			self.prev_layers.append(prev_layer)

	def define_this_layer(self,prev_layer):
		if type(prev_layer)==list:
			for prev_l in prev_layer:
				prev_l.next_layers.append(self)
		else:
			prev_layer.next_layers.append(self)	

	def call_and_store(self,x):
		self.res=self.call(x)
		return self.res

	def get_reses(self,layer_list):
		if len(layer_list)==1:
			return layer_list[0].res
		else:
			return [l.res for l in layer_list]

	def calc(self,layers_list=None):
		for l in self.prev_layers:
			if l.res is None:
				if layers_list:
					layers_list.append(l)
				l.calc(layers_list)

		self.call_and_store(self.get_reses(self.prev_layers))

class Input(layer):
	def __init__(self,input_shape):
		super().__init__()
		self.input_shape=input_shape
		self.compute_output_shape()

	def compute_output_shape(self):
		self.output_shape=self.input_shape

	def init_weights(self):
		pass

	def call(self,x):
		return x

	def get_weights(self):
		return []

	def set_weights(self,weights):
		pass

class Add(layer):
	def __init__(self):
		super().__init__()

	def compute_output_shape(self):
		self.output_shape=self.prev_layers[0].output_shape

	def call(self,x):
		x0,x1=x
		return x0+x1

	def init_weights(self):
		pass

	def get_weights(self):
		return []

	def set_weights(self,weights):
		pass


class Dense(layer):
	def __init__(self,n_uints,activation='linear',kernel_init='uniform',bias_init='zeros'):
		super().__init__()
		self.n_uints=n_uints
		self.activation=activation

		if type(activation)==str:
			self.activation_func=activation_dict[activation]
		else:
			self.activation_func=activation

		if type(kernel_init)==str:
			self.init_weight_class_kernel=init_dict_classes[kernel_init]()
		else:
			self.init_weight_class_kernel=kernel_init

		if type(bias_init)==str:
			self.init_weight_class_bias=init_dict_classes[bias_init]()
		else:
			self.init_weight_class_bias=bias_init

	def compute_output_shape(self):
		self.output_shape=(self.n_uints,)

	def init_weights(self):
		assert len(self.prev_layers)==1
		assert len(self.prev_layers[0].output_shape)==1
		self.kernel_weights=self.init_weight_class_kernel.create_weights(size=(self.prev_layers[0].output_shape[-1],self.n_uints))
		self.bias_weights=self.init_weight_class_bias.create_weights(size=self.output_shape)

	def call(self,x):
		return self.activation_func(np.dot(x,self.kernel_weights)+self.bias_weights)

	def get_weights(self):
		return [self.kernel_weights,self.bias_weights]

	def set_weights(self,weights):
		kernel_weights,bias_weights=weights
		self.kernel_weights=kernel_weights
		self.bias_weights=bias_weights






