import numpy as np
np.random.seed(0)
import gym
from agents import EV_agent
import threading
import multiprocessing
import hashlib
from tqdm import tqdm
import time
def infrange(start=0,end=None,tqdm_bar=False,**kwargs):
	i=start
	if tqdm_bar:
		tqdm_bar=tqdm(total=end-start if end else None,**kwargs)
	while True:
		yield i
		if tqdm_bar:
			tqdm_bar.update(1)
		i+=1
		if end==i:
			if tqdm_bar:
				tqdm_bar.close()
			break

def generator(env,agent,max_t=None,render=False,**kwargs):
	max_t=max_t if max_t else np.inf
	for i_episode in infrange(start=1,**kwargs):
		observation = env.reset()
		obs=[]
		acts=[]
		rewards=[]
		for t in infrange():

			obs.append(observation)
			if render:
				env.render()
				#time.sleep(0.05)

			action = agent.predict(obs,**kwargs)

			acts.append(action)
			observation, reward, done, info = env.step(action)
			rewards.append(reward)

			if done:
				yield i_episode,obs,acts,rewards
				obs=[]
				acts=[]
				rewards=[]
				break


def play(env,agent):
	observation = env.reset()
	rewards=[]
	obs=[]
	for t in infrange():

		obs.append(observation)
		action = agent.predict(obs)
		#env.render()
		observation, reward, done, info = env.step(action)
		rewards.append(reward)

		if done:
			return sum(rewards),t

class playinthreads:
	def __init__(self,n_workers,n_trais,get_env_func,get_agent_func):
		self.n_workers=n_workers
		self.n_trais=n_trais
		self.get_env_func=get_env_func
		self.res_queue=multiprocessing.Manager().Queue()
		self.agents_queue=multiprocessing.Manager().Queue()
		for i in range(self.n_workers):
			agent=get_agent_func(i)
			agent.edit_weights()
			self.agents_queue.put(agent)		
		
		self.run_threads()

	def worker_func(self,i):
		env=self.get_env_func(i)
		t1=time.time()
		while True:
			agent=self.agents_queue.get()
			for i in range(3):
				try:
					score=self.get_score(env,agent)
					self.res_queue.put((score,agent))	
					break
				except Exception as e:
					print('error: ',str(e))				


	def run_threads(self):
		threads=[None]*self.n_workers
		
		for i in range(self.n_workers):
			t=multiprocessing.Process(target=self.worker_func,args=(i,))
			t.start()
			threads[i]=t

	def get_score(self,env,agent):

		res=[]
		for itera in range(self.n_trais):
			score,nep=play(env,agent)
			res.append(score)
		return np.mean(res)

	def run(self,best_weights,i):
		t1=time.time()
		score,agent=self.res_queue.get()
		weights=agent.copy_weights()
		agent.load_weights(best_weights)
		#agent.set_learning_rate(agent.base_learning_rate*(0.9999999**i))
		agent.edit_weights()
		self.agents_queue.put(agent)
		return score,weights

		#print(hashlib.md5(str(best_weights).encode()).hexdigest())

class print_m:
	def __init__(self,format_t,print_each=1):
		self.print_each=print_each
		self.format=format_t
		self.mem=[]
		self.t1=time.time()
	def print(self,*args):
		self.mem.append(args)
		timenow=time.time()
		if (timenow-self.t1)>=self.print_each:
			self.t1=timenow
			print(self.format.format(*np.mean(self.mem,axis=0).tolist()))
			self.mem=[]


#envn='CartPole-v0'
#envn='LunarLander-v2'
envn='BipedalWalker-v2'

import gym_wrappers
def get_env_func(i):
	env=gym.make(envn)
	env.seed(i)
	env._max_episode_steps = 2000
	return env

def get_agent_func(i):
	return EV_agent(env,learning_rate=5,normlize=False)


env=get_env_func(12345)
my_agent=get_agent_func(12345)
best_weights=my_agent.copy_weights()
best_score=-np.inf
best_score_gen=-np.inf
res=[]
try_in_each_model=20
i=0
import psutil

if 0:
	pit=playinthreads(8,5,get_env_func,get_agent_func)
	printm=print_m(
'i_episode={:.0f}, score={:.5f}, best_score={:.5f}, best_score_gen={:.5f}, speed={:.2f}ep/s',
				   print_each=1)

	gen_per_ep=200
	gen_models={}
	t1=time.time()
	for i_episode in infrange(start=1):
		score,weights=pit.run(best_weights,i_episode)
		gen_models[score]=weights
		if score>best_score:
			best_score=score
			#best_weights=weights
			np.save(envn,best_weights)
		if i_episode%gen_per_ep==0:
			if psutil.virtual_memory().percent>=97.0:
				exit()
			best_score_gen=max(gen_models)
			best_weights=gen_models[best_score_gen]
			gen_models={}

		printm.print(i_episode,score,best_score,best_score_gen,1/(time.time()-t1))
		#print(score)
		t1=time.time()
else:
	from gym import wrappers
	#env = wrappers.Monitor(env, 'videos_2/',force=True,video_callable=lambda x:x+1)
	my_agent.load_weights(np.load(envn+".npy"))
	for i_episode,obs,acts,rws in generator(env,my_agent,render=True):
		print(sum(rws))

