import math
import random
import sys
import numpy as np
import operator
import pickle

"""<<<Create a generic Agent class>>>"""

class Agent(object):

	def __init__(self, env, epsilon = 0.8, gamma = 0.99, alpha = 0.01):
		self.env = env
		self.q_table = {}
		
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		
		self.penalties = []
		self.reward = 0
		self.cumulative_reward = 0.0
		self.count = 0

	def reset_reward(self):
		self.reward = 0

	def get_current_state(self, snakelist):
		return self.env.get_state(snakelist)

	def manhattan_dist_target(self):
		appleX, appleY = self.env.get_apple_position()
		headX, headY = self.env.get_head_position()
		return abs(headX - appleX) + abs(headY - appleY)

	def choose_action(self, snakelist):
		max_q = 0

		self.state = self.env.get_state(snakelist)

		if not self.state in self.q_table:
			self.q_table[self.state] = {action : 0 for action in self.env.valid_actions}

		action = random.choice(self.env.valid_actions)

		# Implement epsilon-greedy selection from Q table (exploitation) vs random choice (exploration)
		# Above we initialised the Q table to zero for unseen states and chose a random action.
		# Now, if the Q table dictionary for the current state has only 1 unique value then we can conclude this is a newly seen 
		# state since everything will be 0. In this case, we use the random behaviour. However, if it is previously seen, we will
		# behave greedily and choose action that maximises estimated reward.
		# The behaviour described above happens if we generate a float above epsilon. As training proceeds, epsilon is steadily
		# increased meaning we should become more exploitative. However, this relies on training long enough to see every state.
		if np.random.random() > self.epsilon:
			if len(set(self.q_table[self.state].values())) == 1:
				pass
			else:
				#argmax key (action) across dictionary values (q values)
				action = max(self.q_table[self.state].items(), key = operator.itemgetter(1))[0] 
				#print(self.q_table[self.state])
				#print(action)
		return action

	


	def update(self, action, reward, snakelist):
		""" At the point when update is called in the main code, the action has already been taken
			activate the print statements below to make this manifest"""
		#print(self.count)
		#print("Current state: {}".format(self.state))
		self.cumulative_reward += reward

		self.next_state = self.env.get_state(snakelist)

		#print("Action: {}".format(action))
		#print("Future state: {}".format(self.next_state))

		# Add any unknown state/action pairs to q_table
		if self.next_state not in self.q_table:
			self.q_table[self.next_state] = {action : 0 for action in self.env.valid_actions}

		# Get q_value for action that we just took
		old_q_value = self.q_table[self.state][action]

		# Identify the highest q value of possible next actions (we have greedy policy)
		next_max = max(self.q_table[self.next_state].values())

		# Calculate the q_value for this maximising action
		new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * next_max)
		self.q_table[self.state][action] = new_q_value
		#print(len(self.q_table))
		#if random.randrange(0, 5000) == 5:
		#print("Agent update: state = {}, action = {}, reward = {}".format(self.state, action, reward))






""" <<<TO DO>>>
1. run method needs to be able to save and have replays						   DONE
"""