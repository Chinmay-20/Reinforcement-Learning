#this is a very basic code
#further environment could be extremely complicated and agent could be large NN that implements latest RL algo
from random import random
import random
class Environment:
	def __init__(self):
		self.steps_left=10  #initialize its internal state.
		#state is a counter that limits number of time steps that the agent is allowed to take to interact with environment
		
	def get_observation(self): #this method is supposed to return current environment's observation to agent
		return [0.0,0.0,0.0]
	#-> List[float] it is Python type annotations which were introduced in Python 3.5
	# observation vector is always zero as environment basically has no internal state.
	
	def get_actions(self): #this method allows agent to query set of action it can execute. set of actions doesn't change but some actions can become impossible in different states(eg tic-tac-toe game)
		return [0,1] #in our example there are only two actions that agent can carry out which are encoded with integers 0 and 1.
		
		
	def is_done(self)->bool: #this method signaled end of agent
		return self.steps_left==0
	
	def action(self,action:int) -> float: #action method is central piece in environment's functionality. It does two things handles agent's action and return reward for this action. In our eg reward is random and action is discarded
		if self.is_done():
			raise Exception("Game is over")
		self.steps_left -= 1
		return random.random()
		
class Agent:
	def __init__(self):
		self.total_reward=0.0
		
	def step(self,env:Environment):
		current_obs=env.get_observation()
		actions=env.get_actions()
		reward=env.action(random.choice(actions))
		self.total_reward += reward
#step function accepts environment instance and allows agent to 1)observe environment 2)make a decision about action to take based on observations 3)submit action to environment 4)get the reward for current step
#in our eg agent ignores observations obtained during decision making process on which action is supposed to be taken. instead every action is selected randomly

if __name__=="__main__":
	env=Environment()
	agent=Agent()
	
	while not env.is_done():
		agent.step(env)
		
	print("Total reward got: %.4f"%agent.total_reward)
