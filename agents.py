import numpy as np
import gym
from mynn import layers as ml
from mynn.models import Model
import copy


class EV_agent:
	def __init__(self,env,**kwargs):
		self.env=env
		self.base_learning_rate=kwargs.pop("learning_rate",1)
		self.learning_rate=self.base_learning_rate
		self.resize_factor=kwargs.pop("resize_factor",1)
		self.is_decolor=kwargs.pop("is_decolor",False)
		self.normlize=kwargs.pop("normlize",False)
		self.norm_obs_range=(-1,1)
		env_input_shape=env.observation_space.shape
		self.input_shape=env_input_shape
		if len(env_input_shape)==2:
			self.input_type='2DS'
		elif len(env_input_shape)==3:		
			self.input_type='2DM'
		elif len(env_input_shape)==1:
			self.input_type='1D'
		else:
			raise ValueError("the observation_space type is not recognized")

		if self.is_decolor and self.input_type=='2DM':
			self.input_shape=self.input_shape[:2]+(1,)

		if self.input_type[:2]=='2D':

			shape=self.input_shape
			self.input_shape=(shape[0]//self.resize_factor,shape[1]//self.resize_factor)+shape[2:]#(w//self.resize_factor,h//self.resize_factor,c)

		if type(env.action_space)==gym.spaces.discrete.Discrete:
			self.output_type='Discrete'
		elif type(env.action_space)==gym.spaces.box.Box:
			self.output_type='Box'
		else:
			raise ValueError("the action_space type is not recognized")

		if self.output_type=='Discrete':
			self.output_shape=(env.action_space.n,)
			self.output_activation='softmax'
		elif self.output_type=='Box':
			self.output_shape=env.action_space.shape
			is_low_negone=(env.action_space.low==(-1*np.ones_like(env.action_space.low))).all()
			is_high_posone=(env.action_space.high==(+1*np.ones_like(env.action_space.high))).all()
			if is_low_negone and is_high_posone:
				self.output_activation='tanh'
			else:
				self.output_activation='linear'


		self.norm_indexes_obs=np.where((np.abs(env.observation_space.low)+np.abs(env.observation_space.high))!=np.inf)
		self.low_obs=env.observation_space.low[self.norm_indexes_obs]
		self.high_obs=env.observation_space.high[self.norm_indexes_obs]-self.low_obs
		self.model=self.create_model()
		self.i=0

	def preprocess_obs(self,obs):
		if self.input_type[:2]=='2D':
			p_obs=[]
			for ob in obs:
				pimg=cv2.resize(ob,(self.input_shape[2],self.input_shape[1]))
				if self.is_decolor:
					pimg=np.expand_dims(np.mean(pimg,axis=2),axis=2)
				p_obs.append(pimg)
			p_obs=np.array(p_obs)
		else:
			p_obs=obs

		if self.normlize:
			p_obs=np.array(list(map(self.normlize_obs,p_obs)))

		return p_obs

	def normlize_obs(self,ob):
		ob[self.norm_indexes_obs]=(((ob[self.norm_indexes_obs]-self.low_obs)/self.high_obs)*(self.norm_obs_range[1]-self.norm_obs_range[0]))-self.norm_obs_range[1]
		return ob

	def create_model(self):
		li=ml.Input(input_shape=self.input_shape)
		l=li
		for n_units in [16,8]:
			l=ml.Dense(n_units,activation='elu')(l)
		l=ml.Dense(self.output_shape[0],activation=self.output_activation)(l)
		return Model(li,l)

	def edit_weights(self):
		weights=self.model.get_weights()
		for layer_index in range(len(weights)):
			for weight_index in range(len(weights[layer_index])):
				w=weights[layer_index][weight_index]
				r=np.random.uniform(low=-1,high=1,size=w.shape)
				r=r*self.learning_rate
				w+=r
				weights[layer_index][weight_index]=w
		self.model.set_weights(weights)

	def predict(self,inputs):
		inputs=inputs[-1]
		pred=self.model.predict(inputs)
		if self.output_type=='Discrete':
			pred=np.random.choice(np.arange(self.output_shape[0]),p=pred)
		return pred

	def copy_weights(self):
		return self.model.get_weights()

	def load_weights(self,weights):
		self.model.set_weights(weights)

	def decay(self,r):
		self.i+=1
		#lr=self.base_learning_rate-((self.base_learning_rate/5000)*self.i)
		#self.learning_rate=max(lr,0.05)
		#self.learning_rate=self.base_learning_rate*(r**self.i)

	def set_learning_rate(self,r):
		self.learning_rate=r
