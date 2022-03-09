import gym

if __name__=="__main__":
	env=gym.make("CartPole-v0")
	total_reward=0.0
	total_steps=0
	obs=env.reset() # we reset environment to obtain first observation
	
	while True:
		action=env.action_space.sample() #sampled random action
		obs,reward,done,_=env.step(action)
		total_reward+=reward
		total_steps+=1
		if done:
			break
			
	print("Episode done in %d steps, total reward %.2f"%(total_steps,total_reward))
	
#https://stackoverflow.com/questions/56904270/difference-between-openai-gym-environments-cartpole-v0-and-cartpole-v1
#https://engineering.purdue.edu/DeepLearn/pdf-kak/Reinforcement.pdf
#https://www.mathworks.com/help/reinforcement-learning/ug/create-predefined-control-system-environments.html
#https://www.youtube.com/watch?v=JNKvJEzuNsc
#https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
