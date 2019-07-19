import numpy as np
import copy
import pickle
class Model:
	def __init__(self,input_layers,output_layers):
		self.input_layers=[input_layers] if type(input_layers)!=list else input_layers
		self.output_layers=[output_layers] if type(output_layers)!=list else output_layers
		layers_list=[]
		layers_list.extend(self.output_layers)
		for input_layer in self.input_layers:
			input_layer.call_and_store(np.empty(input_layer.input_shape))
		for output_layer in self.output_layers:
			output_layer.calc(layers_list)
		layers_list.extend(self.input_layers)
		self.layers_list=layers_list[::-1]
		self.clear_prev_res()

	def clear_prev_res(self):
		for layer in self.layers_list:
			layer.clear_res()

	def predict(self,inputs):
		if type(inputs)!=list:
			inputs=[inputs]
		assert len(inputs)==len(self.input_layers)
		for input_arrays,input_layer in zip(inputs,self.input_layers):
			input_layer.call_and_store(input_arrays)

		for output_layer in self.output_layers:
			output_layer.calc()		

		res=[output_layer.res for output_layer in self.output_layers]
		self.clear_prev_res()
		return res[0] if len(res)==1 else res

	def get_weights(self):
		return [[w.copy() for w in l.get_weights()] for l in self.layers_list]

	def set_weights(self,weights):
		for l,weight in zip(self.layers_list,weights):
			l.set_weights(weight)



if __name__ == '__main__':
	
	from layers import *
	li=Input(input_shape=(3,))
	l=li
	l1=Dense(3,activation='elu',kernel_init='ones',bias_init='zeros')(l)
	l=Dense(3,activation='linear',kernel_init='ones',bias_init='ones')(l1)
	l=Add()([l1,l])
	l=Dense(3,activation='elu',kernel_init='ones',bias_init='zeros')(l)
	x=np.array([-1,-1,-1],dtype=np.float32)
	#li.call_and_store(x)
	model=Model(li,l)
	model2=model.copy()
	model2.output_layers[0].bias_weights=np.array([1,1,1,])
	model.set_weights(model2.get_weights())
	print(model.get_weights())
	print(model2.get_weights())
	#l.calc()
	#print(l.res)
	#print(model.get_first_layer(l))
	print(model.predict(x))