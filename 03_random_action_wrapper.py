#in this program we will intervene stream of actions sent by agent and replace 10% current action with random one. this technique is powerful for understanding exploration/exploitaton dilemma
#by issuing random actions we make our agent explore environment and from time to time drift away from beaten track its policy

import gym
from typing import TypeVar
import random

Action=TypeVar('Action')

class RandomActionWrapper(gym.ActionWrapper): #here we initialized wrapper bu calling parent's init method and saving epsilon
	def __init__(self,env,epsilon=0.1):
		super(RandomActionWrapper,self).__init__(env)
		self.epsilon=epsilon
		
	def action(self,action:Action) -> Action: #this is the method we need to override from parent's class to tweak agents action.
	#every time we roll the die and with probability of epsilon we sample a random action from action space and return it instead of action the agent has to sent us.
		if random.random() < self.epsilon:
			print("Random!") #we print message every time we replace the action to verify our action is working
			return self.env.action_space.sample()
		return action
		
if __name__=="__main__":
	env=RandomActionWrapper(gym.make("CartPole-v0")) #create normal cartpole envionment and from here we use wrapper as a normal Env instance instead of orginal CartPole because wrapper class inherits env class and exposes same interface we nest our wrapper in any combination we want.
	
	obs=env.reset()
	total_reward=0.0
	while True:
		obs,reward,done,_=env.step(0)
		total_reward+=reward
		if done:
			break
			
	print("Reward got: %.2f"%total_reward)
	
