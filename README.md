

We have to write a policy to detect errors in furniture.
What will be the observations in our project? Our observations will be continuous.
What is the environment in our case?
03_random_class_wrapper code important for exploration vs exploitation





Wikipedia
It does not need labeled input/output pairs to be presented.
It does not need sub optimal actions to be explicitly corrected.

The main focus is to use current knowledge to tackle new situations.
Partially supervised RL algo combines both supervised and RL algo

The environment is set in form of Markov Decision Process(MDP)
MDP is a time stochastic control process and provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under control of the decision maker.



Deep Learning Basics: Introduction and Overview
deeplearning.mit.edu

Deep learning is representation learning ie feature learning

Machine learning : input + feature extraction + classification + output
Deep Learning : input + (feature extraction + classification) + output

Humanoid and autonomous vehicles are not data driven learning instead they are model based learning

Recurrent neural networks to predict the future.

So we design an RL agent that optimizes for rewards and that RL agent has nothing to do with finishing race, they get much more points by focusing on rewards


https://www.youtube.com/watch?v=J9JZyyPCJcQ&list=PLImtCgowF_ES_JdF_UcM60EXTcGZg67Ua&index=1

Goal of AI : to understand general principles behind human intelligence. And to stimulate them in machines

Machine learning is one way to solve AI. It takes a learning approach to solve AI. ML is all about designing algo that can be learned through experience.

RL is learning through interaction. In RL there is an agent and all other things except agent are environment. Agent receives an observation from the environment. And after observing it will take action. It not only receives observation but also receives rewards 
What is the purpose of this interaction?
The main objective is to learn how to map situations to action. So to maximize a numerical record
Eg there is a trash collecting robot. It receives a reward of +1 whenever he collects the trash and reward of 0 for the rest of the kind. And reward of -100 if it runs out of battery.
The robot should collect trash and on the parallel side he must look at the battery and keep in mind the distance between itself and the charger point. 
 
Lecture 2 (questions in last after 43 min) 
There might be a possibility that a trash collecting robot picks up trash and throws it and again collects it. This might be faulty reward points

Reward hypothesis : That all of what we mean by goals and purposes can be well thought of as maximum of the expected value of cumulative sum of received scalar signal(reward)

Any goal that you want to have can be thought of maximizing expected value
It means that you can construct the reward and maximizing that reward will help the agent to achieve that goal.

Reward-is-enough : Intelligence and its associated liabilities can be understood as subserving the max of reward by an agent acting in its environment.
Reward-is-enough tells us that any goal you want to construct, construct a reward function for that and maximizing is enough, you can do all of intelligence by having reward structure and any agent who will be maximizing reward structure is implementing intelligence

In supervised learning the goal is to train a function from given data such that x->y and for new x accurately predict y. It is learning from data. In SL we only predict. SL is having labels

In unsupervised learning you have only x in the dataset. The goal is to find some structure in data. Eg clustering . if i had a bunch of news articles then it will cluster them according to data but i will not cluster them . if i knew the clusters then it will be supervised learning

Reinforcement learning : RL is all about learning from interaction. In RL we predict and control the environment. RL is trial and error process. In RL there is an environment and it is giving some observations. These observations are given to agents. An agent makes the prediction. This action is sent to the environment again. The environment gives you new observations and rewards. And this cycle continues. The main difference between RL and SL,UL is back and forth interaction between agent and environment. Second major difference is the notion of time in RL is temporary decision making ie u are acting for actions one after the other. 

Practical usage. Deep mind used RL to control power usage in the data  center. Thy were able to reduce 30 - 40 % usage

The actual challenge is this loop. When an agent takes action 1. It is not only affecting the immediate reward but all the future rewards we will see. And that’s because of the loop. Because actions decide the next state. 

Implications of closed loop (implication : conclusion that cannot be explicitly stated)
The goal of an agent is not only to get immediate reward but it is to maximize cumulative reward i.e. get good rewards for life. It states that sometimes you have to go for minimum rewards that will lead you to higher rewards. Eg in chess you have to sacrifice a pawn to get a queen.

Every action has immediate and cumulative reward. 

The problem of crediting which action is responsible for this good reward or bad reward is known as credit assignment.
 Taking the loop into consideration there are two views, forward view and backward view. In the forward  view reward is a cumulative effect. And in the backward view this cumulative effect is a problem known as credit assignment.

We need to figure out which action leads to which reward

Challenges in building RL algo:
Exploration vs exploitation : doing trial and error can sometimes be costly. Exploit : what it has already experienced in order to obtain reward. Explore : in order to make better action selection in future. If you are always exploring then u might not get maximum reward. And again if you are always exploiting then u might not get maximum reward. So we must have a balance between exploring and exploiting. And this is the hardest and unique problem. 

Do we allow the agent to learn the task before putting it into the real world?
Ans: The actual thing is everything starts in the real world. You are learning through interactions. But some learning should happen offline under safe RL. (amazon automated drones)

There is no dataset in RL. We are learning through interaction. 

Lecture 3:

There are two types of feedback
Instructive feedback : tells the agent what to do exactly ( gives result + tells exact output). Used in SL
Evaluative feedback : it just evaluates the action and doesn't tell what to do.( only gives result). Used in RL. And RL is all about this evaluative feedback

Key characteristics of RL problem
Learning to act in many situation
Delayed rewards
Exploration vs exploitation dilemma 

To understand exploration vs exploitation problems we will use the concept of immediate RL.
In this we will only consider one situation and we will give immediate rewards. Now it is completely an exploration vs exploitation problem.

(9 minute)	
Immediate RL / K-armed bandit problem is simplest possible RL setting
In this reward chosen from stationary probability distribution that depends on your action.

Objective is to maximize expected total reward over some time period. Objective is to find best arm as quickly as possible so that we can exploit it to get maximum rewards

It is simple because agent sees same set of actions of all times. And also rewards are immediate



Deep reinforcement learning by mohit sewak

Chapter 1
AI is study and design of rational agents which could act humanly. The agent should posses ability to demonstrate thought process and reasoning, intelligent behavior, success in terms of human performance

Here an agent interacts with environment to learn to take best possible action under given state. The action changes state of environment from S to St+1 and environment generates reward for agent. Same procedure is repeated. Over a period of iterations the agent tries to improve upon its decision of which is best action.

The role of environment is to present agent with different probable states.

Reward is function of both action and state and not action alone. This means same action could receive a different reward under different states.

The role of agent is to take action that pays max reward.



Deep reinforcement learning hands on

Normal ML/DL applications have a hidden time dimension. Eg model trained to classify dog and cat. But after doing makeover of dog it is giving inappropriate results. 	
Supervised : we learn from known answers provided by “ground truth”
Unsupervised : to learn some hidden structure of dataset at hand.

RL lies between full supervision and complete lack of predefined labels. It uses deep neural networks for function approximation, stochastic gradient descent, backpropagation to learn data representation and applies them in different way

RL depends on the agent's behavior and the result of this behavior. So if an agent does wrong behavior and continues the same it will not get a larger reward. This is first problem
Our RL model needs to not only exploit, but also explore the environment. 
This exploration / exploitation dilemma is second problem
Third problem can be rewards can be delayed. 

The term reinforcement comes from fact that reward obtained by agent should reinforce its behavior in positive or negative way
Reward does not reflect success achieved by agent so far.

Environment is everything outside agent. Agents communication with environment is limited to rewards, actions, observations, 

Actions are things agent can do. Eg. move pawn one space forward or fill tax form in for tomorrow morning.
There are two types of actions:
Discrete : it is a finite set of mutually exclusive things agents can do. Eg move left or right.
Continuous : it has some value attached to them. Eg cars action turn the wheel as value of angel matters.

Observations form a second information channel while reward forms first.
Observations tell agent what’s going around the agent

Reward is main force that drives agents learning process.

Theoretical foundations
Markov process -> (with rewards) markov reward process -> (extra actions) Markov decision process

Markov process:
There is only agent that can only observe and cannot influence system but can only watch states changing.
State space is all possible states. For MPs states should be finite.
Observations must form chains

Markov property means future state dynamics from any state have to depend on this state only. Main point of Markov property is that MP requires states of the system to be distinguishable from each other and unique.
Markov reward process 
If gamma is zero the return will be immediate reward without any subsequent state and will correspond to short - sightedness
If gamma is 1 return will be sum of all subsequent rewards

Policy 
It is some set of rules that controls agents behavior
Different policies can give us different amounts of returns, so it is important to find a good policy.

Policy is defined as probability distribution over actions for every possible state.
Another useful notion is that if our policy is fixed and not changing, then our MDP
becomes a Markov reward process, as we can reduce the transition and reward
matrices with a policy's probabilities and get rid of the action dimensions.



Chapter 2

Boilerplate code means a piece of code that can be used again and again.

8/3/2021
In practice an agent is a piece of code that implements some policy. This policy decides what action is needed  at every time step, given our observations
The environment has the responsibility of providing observations and giving rewards. The environment changes its state based on the agent's action.

EG
We define an environment that will give agents random rewards for a limited number of steps regardless of agent actions. This eg will allow us to focus on specific methods in both environment and agent classes

Code in rl1.py in deep reinforcement learning.

External libraries we will use opencv, numpy, gym (RL framework that has various environments that can be communicated with in a unified way), PyTorch(deep learning library), PyTorch Ignite(high level tools on top of PyTorch) , PTAN (extension to gym to support modern deep RL methods and building blocks

The OpenAI Gym API
Main goal of Gym is to provide rich collection of environments for RL experiments using a unified interface. There are several methods and fields that provide required information about its capabilities. Gym supports both discrete and continuos actions as well as combination. Gym also has step method to execute an action, which return current observation, reward, indication that episode is over. Another method called reset, return environment to its initial state and obtains first observation

Components of environment 
The action space
Agent can take continous, discrete or both agents
Discrete are fixed set of things that agent can do. For eg. move left, right, up, down, press or release button. Main characteristic of discrete action is only one action from finite set is possible
Continuos action has a value attached to it. Eg steering wheel can be turned at specific angle(-720 to 720) , accelerator pedal can be pressed with different level of force(0 to 1). There are value boundaries the action could have.
To support multiple actions Gym defines a special container class that allows nesting of several spaces into unified action

The observation space
Observations are information piece provided by environment with timestamp. Observations can be bunch of numbers or several multidimensional tensors. Observation can also be discrete much like action spaces. Eg lightbulb which can be in two states on or off.

The basic abstract class space includes two methods
sample() : this returns a random sample from space
contains(x) : checks whether argument x belongs to space’s domain
This methods are abstract and reimplemented in each subclass box, discrete, tuple

Discrete class represents set of items numbered from 0 to n-1. It has only one field describes count of items. Discrete(n=4) can be used for space of four directions [left,right,up,down]

Box class represents n dimension tensor of rational number with intervals [low,high] .Box could be an Atari screen observation which is an RGB (red, green, and blue) image of size 210×160: Box(low=0, high=255, shape=(210, 160,3), dtype=np.uint8) . In this case, the shape argument is a tuple of three elements: the first dimension is the height of the image, the second is the width, and the third equals 3 , which all correspond to three color planes for red, green, and blue, respectively. So, in total, every observation is a three-dimensional tensor with 100,800 bytes

Tuple class combines several space class instance together. This enables us to create action and observation spaces of any complexity that we want. This is rarely used.

The environment is represented by Env class in Gym which has following members
Action_space : provides a specification for allowed actions in environment
Observation_space : specifies observations provided by environment
reset() : resets the environment to initial state returning initial observation vector.
step() : allows agent to take action and return information about outcome of action - next observation, local reward, end-of-episode flag
render() : allows us to obtain observation in human-friendly form

reset() method has no arguments, it instructs an environment to reset into its initial state and obtain initial observation. Call reset() after creation of environment. The value returned by this method is first observation of environment.

step() is central piece in environment it does several things. 
Telling environment which action we will execute on next step.
Getting new observation from environment after action
Getting reward the agent gained with this step
Getting indication that episode is over
Action is the only argument to this method and rest are returned by step() method. The return is tuple of four elements (observation, reward, done and info)
Observation : numpy vector or matrix with observation data
Reward: float value of reward
Done: boolean indicator which is true when episode is over
Info : this could be anything, we just ignore in general RL methods

Creating environment
Every environment has a unique name of EnvironmentName-vN form where N is number used to distinguish between different versions of same environment
To create an environment gym package provides make(env_name) function name in string format

There are 154 unique environment can be divided into several group
One of them is MuJoCo simulator used for several continuous problems. Parameter tuning - this is RL being used to optimize NN parameters.

We create new environment cartpole. The trickiness is that this stick tends to fall right or left and you need to balance it by moving the platform to the right or left on every step.
Observation is four floating point numbers containing information about x-coordinate  of stick center of mass, speed, angle to platform, and angular speed.
Our problem is to balance stick without knowing parameters by just balancing stick.

The reward in this environment is 1 and is given on every timestamp. The episode continues until stick falls so to get more accumulated reward we need to balance.

WE ALWAYS NEED TO RESET NEWLY CREATED ENVIRONEMENT

9/8/22
Most of the environments in gyms have a “reward boundary” which is the average reward that agents should gain during 100 consecutive episodes to solve the environment.
For cartpole this boundary is 195 which means that on average agent must hold stick for 195 times steps or longer.

Wrapper class : when we want to wrap existing environment and add some extra logic for doing  something then wrapper class is used. Wrapper class inherits Env class. Its constructor accepts only instance of environment class to be wrapped. 
We need to redefine step() or reset() function.
There are subclasses of wrapper that allow only filtering of only specific portions of information.
Observation wrapper : redefine obs method of parent. Obs argument is an observation from wrapped environment. This method will return observation that will be given to agent.
Reward wrapper: exposes reward method which can modify reward value given to agent
Action wrapper: override action (act)  method which can tweak the action passed to wrapped environment by agent

To refer the code for wrapper classes refer code 03_random_action_wrapper.py

Monitor 
It can write information about your agent’s performance in file, with an optional video recording 
