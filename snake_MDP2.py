import pygame
import time
import numpy as np
import random
import optparse
from agent import Agent 
from pprint import pprint
import math
import matplotlib.pyplot as plt
import pylab
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])


"""HYPERPARAMETERS"""
DELAY = 150
GRIDLINES_ON = True
EPISODES = 5000
FPS = 1
TRAINING_SAVE = False
FIXED_TARGET = True

#dimenstions of the window
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600
BLOCK_SIZE = 60






#Choose to play yourself or with agent (flag -m for machine and -p for person)
parser = optparse.OptionParser()
parser.add_option("-m", "--machine", action="store_true", dest="user_input", help="AI to play")
parser.add_option("-p", "--person", action="store_false", dest="user_input", help="Person to play")
(options, args) = parser.parse_args()
# Exit with error 1, if left over arguments are found.
if len(args) is not 0:
	sys.stderr.write("Error passing arguments. Leftover arguments found : %s"%(",".join(args)))
	sys.exit(1)

AI = options.user_input


# Initailize PyGame object
pygame.init()


""" <<< LOGGING INFO TO DISPLAY>>> """
font = pygame.font.SysFont("ubuntu", 22)
largefont = pygame.font.SysFont(None, 40)

def display_score(score, reward):
	text = largefont.render("Score: {}                           Agent reward: {}".format(str(score),str(reward)), True, pink)
	gameDisplay.blit(text, [10,DISPLAY_HEIGHT+7])

def display_AI_score(count, manhattan_dist):
	text = font.render("Episode {}/{}          L1 Distance To Target = {}".format(str(count+1), EPISODES, int(manhattan_dist)), True, green)
	gameDisplay.blit(text, [10,DISPLAY_HEIGHT+10])

def display_AI_score2(score, reward, top_score):
	text = font.render("Score: {}          Agent Reward: {}          Top Score: {}".format(str(score),str(reward), str(top_score)), True, green)
	gameDisplay.blit(text, [10,DISPLAY_HEIGHT+40])

def display_AI_score3(epsilon):
	text = font.render("Epsilon Greed: {}%          Map Exploration: {}%".format(round(100*(1-epsilon), 2), round((len(coords)/((DISPLAY_WIDTH/BLOCK_SIZE)**2))*100,1)), True, green)
	gameDisplay.blit(text, [10,DISPLAY_HEIGHT+70])

def display_AI_score4(avg_longevity, avg_time_to_target, avg_score):
	text = font.render("Avg Longevity = {}          Avg Time To Target = {}          Avg Score = {}".format(round(avg_longevity, 2), round(avg_time_to_target, 2), round(avg_score, 2)), True, green)
	gameDisplay.blit(text, [10,DISPLAY_HEIGHT+100])


def create_text_object(text, color):
	textSurface = font.render(text, True, color)
	return textSurface, textSurface.get_rect()


def message_to_screen(msg, color, y_displace=0):
	textSurf, textRect =  create_text_object(msg, color)
	textRect.center = (DISPLAY_WIDTH/2), (DISPLAY_HEIGHT/2)+y_displace
	gameDisplay.blit(textSurf, textRect)


""" <<<FUNCTIONS FOR DRAWING ON SCREEN>>> """
#Defining colors (rgb values)
BACKGROUND_COLOR = (0, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
pink = (255, 20, 147)

#set up the display
gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT + 130))
pygame.display.set_caption("AI Snake")
clock = pygame.time.Clock()


def draw_snake(snakelist, block_size):
	if len(snakelist) < 1:
		for x,y in snakelist:
			pygame.draw.rect(gameDisplay, green, [x, y, block_size, block_size])
	else:
		for x,y in snakelist:
			pygame.draw.rect(gameDisplay, green, [x, y, block_size, block_size])
			i = snakelist[-1][0]
			j = snakelist[-1][1]
			pygame.draw.rect(gameDisplay, green, [i, j, block_size, block_size])
			centre = block_size//2
			radius = 3
			circleMiddle = (int(i+10),int(j+15))
			circleMiddle2 = (int(i + 20), int(j+15))
			pygame.draw.circle(gameDisplay, black, circleMiddle, radius)
			pygame.draw.circle(gameDisplay, black, circleMiddle2, radius)

			
def drawGrid(w, rows, surface):
    sizeBtwn = w // rows
 
    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn
 
        pygame.draw.line(surface, (255,255,255), (x,0),(x,w))
        pygame.draw.line(surface, (255,255,255), (0,y),(w,y))


def initialize_random_position(display_width, display_height, block_size):
	x = random.randrange(0, display_width, step=block_size)
	y = random.randrange(0, display_height, step=block_size)
	# x = round(random.randrange(0, display_width - block_size,)/float(block_size))*block_size
	# y = round(random.randrange(0, display_height - block_size)/float(block_size))*block_size
	# print(x, y)
	return x, y


""" <<<DIRECTIONAL INFO>>> """
# Directions
ALLOWED_DIRS = ["LEFT", "RIGHT", "UP", "DOWN"]

opposite_dirs = {'UP' : 'DOWN',
				 'DOWN' : 'UP',
				 'LEFT' : 'RIGHT',
				 'RIGHT' : 'LEFT'}


""" <<<ENVIRONMENT CLASS>>> """
# Environment object needs to be able to receive an action from the snake object, and carry out that action by moving
# the snake on the screen. Importantly, it must also be able to respond to the action by issuing an appropriate reward
# that will be fed back to the agent.
class Environment(object):
	def __init__(self,
		         display_width,
		         display_height,
		         block_size,
		         valid_directions):

		self.world_width = display_width
		self.world_height = display_height
		self.block_size = block_size
		self.lead_x = display_width/2
		self.lead_y = display_height/2
		self.lead_x_change = 0
		self.lead_y_change = 0
		self.valid_actions = valid_directions

		self.highest_score_so_far = -1

		self.appleX, self.appleY = initialize_random_position(self.world_width,
			                                                  self.world_height,
			                                                  self.block_size)

	def act(self, action, snakelist, reward):
		'''
		Given an action, return the reward.
		'''
		#reward -= 0.01 # this is time penalty
		collision = "none"
		is_boundary = self.is_wall_nearby()
		is_near_snake = self.is_snake_nearby(snakelist)

		if self.get_head_position()[0] == 0.0 and action == 'LEFT':
			reward -= 5
			collision = "wall"
		elif self.get_head_position()[0] == float(DISPLAY_WIDTH - BLOCK_SIZE) and action == 'RIGHT':
			reward -= 5
			collision = "wall"
		elif self.get_head_position()[1] == 0.0 and action == 'UP':
			reward -= 5
			collision = "wall"
		elif self.get_head_position()[1] == float(DISPLAY_HEIGHT - BLOCK_SIZE) and action == 'DOWN':
			reward -= 5
			collision = "wall"
		else:
			reward -= 0.1
			self.move(action)
			if self.is_goal_state(self.get_head_position()[0], self.get_head_position()[1]):
				reward += 1 
				collision = "food"
				if not FIXED_TARGET:
					self.new_apple(snakelist)
			elif (self.get_head_position()[0], self.get_head_position()[1]) in snakelist:
				#reward -= 3 * snakeLength / 2
				collision = "snake"
		return reward, collision

	def move(self, direction):
		x_change = 0
		y_change = 0
		
		if direction in ALLOWED_DIRS:
			if direction == "LEFT":
				x_change = -self.block_size
				y_change = 0
			elif direction == "RIGHT":
				x_change = self.block_size
				y_change = 0
			elif direction == "UP":
				x_change = 0
				y_change = -self.block_size
			elif direction == "DOWN":
				x_change = 0
				y_change = self.block_size
		else:
			print("Invalid direction.")

		self.lead_x += x_change
		self.lead_y += y_change
		return self.lead_x, self.lead_y

	def is_wall_nearby(self):
		left, right, up, down = False, False, False, False
		if self.lead_x - self.block_size < 0:
			left = True
		if self.lead_x + self.block_size >= self.world_width:
			right = True
		if self.lead_y - self.block_size < 0:
			up = True
		if self.lead_y + self.block_size >= self.world_height:
			down = True

		return {
			"LEFT":left,
			"RIGHT":right,
			"UP":up,
			"DOWN":down
		}

	def is_snake_nearby(self, snakelist):
		left, right, up, down = False, False, False, False
		for snake_segment in snakelist:
			if (self.lead_x - snake_segment[0]) - self.block_size < 0:
				left = True
			if (snake_segment[0] - self.lead_x) - self.block_size < 0:
				right = True
			if (self.lead_y - snake_segment[1]) - self.block_size < 0:
				up = True
			if (snake_segment[1] - self.lead_y) - self.block_size < 0:
				down = True

		return {
			"LEFT":left,
			"RIGHT":right,
			"UP":up,
			"DOWN":down
		}

	def get_state(self, snakelist):
		environment_state = []
		for x in range(0, DISPLAY_WIDTH, BLOCK_SIZE):
			for y in range(0, DISPLAY_HEIGHT, BLOCK_SIZE):
				is_head = bool((x, y) == self.get_head_position())
				is_body = bool((x, y) in snakelist)
				is_food = bool((x, y) == self.get_apple_position())
				square_state = (is_head, is_body, is_food)
				environment_state.append(square_state)
		return tuple(environment_state)
		
	def get_next_goal(self):
		return (self.appleX, self.appleY)

	def is_goal_state(self, x, y):
		if (x-self.block_size < self.appleX <x + self.block_size  and 
			y-self.block_size < self.appleY <y + self.block_size):
			return True
		return False

	def get_head_position(self):
		return self.lead_x, self.lead_y

	def get_apple_position(self):
		return self.appleX, self.appleY

	def new_apple(self, snakelist):
		while True:
			self.appleX, self.appleY = initialize_random_position(self.world_width, self.world_height, self.block_size)
			if (self.appleX, self.appleY) in snakelist:
				continue
			return self.appleX, self.appleY

	def get_apple_quadrant(self):
		appleX, appleY = self.get_apple_position()
		x, y = self.get_head_position()
		quadrant = 0

		#shift the origin
		appleX -= x
		appleY -= y

		if appleX > 0 and appleY > 0: 
			quadrant = 1
		elif appleX < 0 and appleY > 0:
			quadrant = 2
		elif appleX < 0 and appleY < 0:
			quadrant = 3
		elif appleX > 0 and appleY < 0:
			quadrant = 4
		elif appleX == 0:
			if appleY > 0:
				quadrant = random.choice([1, 2])
			if appleY < 0:
				quadrant = random.choice([3, 4])
		elif appleY == 0:
			if appleX > 0:
				quadrant = random.choice([1, 4])
			if appleX < 0:
				quadrant = random.choice([2, 3])
		return quadrant

	def set_high_score(self, val):
		self.highest_score_so_far = val

	def high_score(self):
		return self.highest_score_so_far

# Initialize the environment	
env = Environment(DISPLAY_WIDTH,
	              DISPLAY_HEIGHT,
	              BLOCK_SIZE,
	              ALLOWED_DIRS)


""" <<<GAME LOOP>>> """
# This depends on whether AI is true (-m flag) or if AI is false (-p flag)
if AI == True:
	agent = Agent(env)

	gameExit = False

	snakelist = []
	snakeLength = 1

	direction = ''
	reward = 0
	old_manhattan_target = 0
	coords = []
	original_epsilon = agent.epsilon
	top_ten_scores = []
	longevity = 0
	avg_longevity = 0
	time_to_target = 0
	avg_time_to_target = 0
	meal_count = 0
	score_occurence = {}
	longevities = []
	mealtimes = []
	avg_score = 0
	avg_scores = []
	reward_list = []
	
	while not gameExit:

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				gameExit = True


		direction = agent.choose_action(snakelist)
		#print(direction)
		# Draw apple and background
		gameDisplay.fill(BACKGROUND_COLOR)
		if GRIDLINES_ON == True:
			drawGrid(DISPLAY_WIDTH, int(DISPLAY_WIDTH / BLOCK_SIZE), gameDisplay)

		apple = env.get_apple_position()

		if direction:
			if snakeLength > 1:
				if direction == opposite_dirs[current_direction]:
					direction = current_direction
			
			pygame.time.wait(200)
			
			reward, collision = env.act(direction, snakelist, reward)



			new_manhattan_target = agent.manhattan_dist_target()

			if new_manhattan_target < old_manhattan_target:
				reward += 0 
			elif new_manhattan_target == old_manhattan_target:
				reward -= 0
			else:
				reward -= 0 

			# Update Q table, increment longevity and time_to_target, update map exploration and reset manhattan target
			agent.update(direction, reward, snakelist)
			longevity += 1
			time_to_target += 1
			(x, y) = agent.env.get_head_position()
			if (x, y) not in coords:
				coords.append((x, y))
			old_manhattan_target = new_manhattan_target

			# Head of the snake
			snake_head = env.get_head_position()
			snakelist.append(snake_head)
			score = snakeLength-1

			# check if the snake hit the wall
			if collision == "wall":
				print(collision)
				#reset score if necessary
				if score > env.high_score():
					print("score: {}. Previous high score: {}".format(score, env.high_score()))
					env.set_high_score(score)
					if len(top_ten_scores) <= 10:
						top_ten_scores.append(score)
						score_occurence[score] = 1
				if score in top_ten_scores:
					score_occurence[score] += 1
				# reset snake
				snakelist = []
				snakeLength = 1
				env.lead_x = 2*BLOCK_SIZE
				env.lead_y = 2*BLOCK_SIZE
				# track the reward for the last episode and increment the episode (agent) count
				reward_list.append(reward)
				reward = 0
				agent.count += 1
				# save if necessary
				if TRAINING_SAVE:
					np.save("training_data/agent-{}-{}-{}".format(str(agent.epsilon), str(agent.reward), str(agent.count)), agent.q_table)
				# Gradually decrement epsilon to the point of maximum greed (hopefully reached optimum policy by end of EPISODES)
				agent.epsilon -= ((original_epsilon)/EPISODES) 
				# keep track of avg score and life expectancy of agent
				avg_longevity = avg_longevity * ((agent.count-1)/(agent.count)) + longevity / agent.count
				if len(score_occurence) > 1:
					avg_score = avg_score * ((agent.count-1)/(agent.count)) + score / agent.count
				if agent.count % 10 == 0:
					longevities.append(avg_longevity)
					mealtimes.append(avg_time_to_target)
					avg_scores.append(avg_score)
				#print(agent.count)
				#print(longevity)
				#print(avg_longevity)
				#print("---")
				longevity = 0

			# normal motion of the snake deletes the old snake tail every time it moves
			if len(snakelist) > snakeLength:
				del(snakelist[0])

			# check if the snake hit itself
			if snake_head in snakelist[:-1] and snakeLength>1:
			 	print(collision)
			 	# reset score if necessary
			 	if score > env.high_score():
			 		print("score: {}. Previous high score: {}".format(score, env.high_score()))
			 		env.set_high_score(score)
			 		if len(top_ten_scores) <= 10:
			 			top_ten_scores.append(score)
			 			score_occurence[score] = 1
			 	if score in top_ten_scores:
			 		score_occurence[score] += 1
			 	# reset snake
			 	snakelist = []
			 	snakeLength = 1
			 	env.lead_x = 2*BLOCK_SIZE
			 	env.lead_y = 2*BLOCK_SIZE
				# track the reward for the last episode and increment the episode (agent) count
			 	reward_list.append(reward)
			 	reward = 0
			 	agent.count += 1
			 	# save if necessary
			 	if TRAINING_SAVE:
			 		np.save("training_data/agent-{}-{}-{}".format(str(agent.epsilon), str(agent.reward), str(agent.count)), agent.q_table)
			 	# Gradually decrement epsilon to the point of maximum greed (hopefully reached optimum policy by end of EPISODES)
			 	agent.epsilon -= ((original_epsilon)/EPISODES)
			 	# keep track of avg score and life expectancy of agent
			 	avg_longevity = avg_longevity * ((agent.count-1)/(agent.count)) + longevity / agent.count
			 	if len(score_occurence) > 1:
			 		avg_score = avg_score * ((agent.count-1)/(agent.count)) + score / agent.count
			 	if agent.count % 10 == 0:
			 		longevities.append(avg_longevity)
			 		mealtimes.append(avg_time_to_target)
			 		avg_scores.append(avg_score)
			 	#print(agent.count)
			 	#print(longevity)
			 	#print(avg_longevity)
			 	#print("---")
			 	longevity = 0


			# check if the snake hit food 
			if collision == "food":
				meal_count += 1
				snakeLength += 1
				avg_time_to_target = avg_time_to_target * ((meal_count-1)/(meal_count)) + time_to_target / meal_count
				time_to_target = 0
				current_direction = direction

			# draw snake, food and log info on screen
			pygame.draw.rect(gameDisplay, red, [apple[0], apple[1], BLOCK_SIZE, BLOCK_SIZE])
			draw_snake(snakelist, BLOCK_SIZE)
			display_AI_score(agent.count, agent.manhattan_dist_target()/30)
			display_AI_score2(snakeLength-1, reward, env.high_score())
			display_AI_score3(agent.epsilon)
			display_AI_score4(avg_longevity, avg_time_to_target, avg_score)
			current_direction = direction
			
			# if training is complete exit the pygame loop, print final Q table and statistical graphs
			if agent.count == EPISODES:
				gameExit = True

				print(agent.q_table.values())

				

				score_dict = {ordinal(i+1) : top_ten_scores[len(top_ten_scores)-i-1] for i in range(len(top_ten_scores))}
				pprint("The top ten scores were: {}".format(score_dict))
				print("Average longevity: {} moves".format(round(avg_longevity, 2)))
				print("Average time to target: {} moves".format(round(avg_time_to_target, 2)))
				print("Average score = {}".format(avg_score))
				print("The distribution of scores was: {}".format(score_occurence))

				e = [10*(i+1) for i in range(len(longevities))]
				l = longevities
				m = mealtimes
				s = avg_scores
				feeding_ratio = int((avg_time_to_target/avg_longevity)* 1000)

				pylab.plot(e, l, '-b', label = 'Avg. Longevity')
				pylab.plot(e, m, '-r', label = 'Avg Time To Target')
				pylab.legend(loc = 'upper left')
				pylab.title('Training Metrics')
				pylab.xlabel('Episodes')
				pylab.ylabel('No. Moves')
				pylab.plot()
				pylab.savefig('episodes={}_100xalpha={}_100xgamma={}_100xepsilon={}__1000xfeeding_ratio={}'.format(EPISODES, int(100*agent.alpha), int(100*agent.gamma), int(100*agent.epsilon), feeding_ratio))
				pylab.show()

				pylab.plot(e, s, '-g')
				pylab.title('Average Score')
				pylab.xlabel('Episodes')
				pylab.ylabel('Score')
				pylab.plot()
				pylab.savefig('episodes={}_100xalpha={}_100xgamma={}_100xepsilon={}__1000xavg_score={}'.format(EPISODES, int(100*agent.alpha), int(100*agent.gamma), int(100*agent.epsilon), int(1000*avg_score)))
				pylab.show()

				score_occurence_keys = score_occurence.keys()
				score_occurence_values = score_occurence.values()
				plt.bar(score_occurence_keys, score_occurence_values)
				score_int = range(min(score_occurence_keys), math.ceil(max(score_occurence_keys))+1)
				plt.xticks(score_int)
				plt.xlabel('Score')
				plt.ylabel('Frequency')
				plt.title('Distribution of Training Scores')
				pylab.savefig('episodes={}_100xalpha={}_100xgamma={}_100xepsilon={}__distribution'.format(EPISODES, int(100*agent.alpha), int(100*agent.gamma), int(100*agent.epsilon), int(1000*avg_score)))
				plt.show()

			

		pygame.display.update()
	

	# Also get the rewards collected
	e = [i for i in range(EPISODES)]
	pylab.plot(e, reward_list, '-g')
	pylab.title('Rewards Collected')
	pylab.xlabel('Episodes')
	pylab.ylabel('Reward')
	pylab.plot()
	pylab.savefig('episodes={}_100xalpha={}_100xgamma={}_100xepsilon={}__reward'.format(EPISODES, int(100*agent.alpha), int(100*agent.gamma), int(100*agent.epsilon), int(1000*avg_score)))
	pylab.show()

	clock.tick(FPS)

# below we handle the single player version of the game - very similar (but simpler) than the above
else:
	gameExit = False

	snakelist = []
	snakeLength = 1

	direction = ''
	reward = 0
	
	while not gameExit:

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				gameExit = True #This flip allows us to manually exit the above while loop and quit the pygame window manually
			if event.type == pygame.KEYDOWN:
		 		if event.key == pygame.K_LEFT:
		 			direction = 'LEFT'
		 		elif event.key == pygame.K_RIGHT:
		 			direction = 'RIGHT'
		 		elif event.key == pygame.K_UP:
		 			direction = 'UP'
		 		elif event.key == pygame.K_DOWN:
		 			direction = 'DOWN'
		
		# Draw apple and background
		gameDisplay.fill(BACKGROUND_COLOR)
		if GRIDLINES_ON == True:
			drawGrid(DISPLAY_WIDTH, int(DISPLAY_WIDTH / BLOCK_SIZE), gameDisplay)
		apple = env.get_apple_position()

		if direction:
			if snakeLength > 1:
				if direction == opposite_dirs[current_direction]:
					direction = current_direction

			pygame.time.wait(DELAY) # a delay is required to make the game playable
			reward, collision = env.act(direction, snakelist, reward)

			# Head of the snake
			snake_head = env.get_head_position()
			snakelist.append(snake_head)
			score = snakeLength-1


			# check if the snake hit the wall
			if collision == "wall":
				if score > env.high_score():
					print("score: {}. Previous high score: {}".format(score, env.high_score()))
					env.set_high_score(score)
				snakelist = []
				snakeLength = 1
				env.lead_x = DISPLAY_WIDTH/2
				env.lead_y = DISPLAY_HEIGHT/2
				reward = 0

			appleX, appleY = apple
			
			
			if len(snakelist) > snakeLength:
				del(snakelist[0])

			#when snake runs into itself
			if snake_head in snakelist[:-1] and snakeLength>1:
				print("snake ran over itself",snakeLength-1)
				if score > env.high_score():
					print("score: {}. Previous high score: {}".format(score, env.high_score()))
					env.set_high_score(score)
				snakelist = []
				snakeLength = 1
				env.lead_x = DISPLAY_WIDTH/2
				env.lead_y = DISPLAY_HEIGHT/2
				reward = 0

			if collision == "food":
				snakeLength += 1
				print("Current reward: {}".format(reward))

			pygame.draw.rect(gameDisplay, red, [apple[0], apple[1], BLOCK_SIZE, BLOCK_SIZE])
			draw_snake(snakelist, BLOCK_SIZE)
			display_score(snakeLength-1, reward)

			current_direction = direction
		
		pygame.display.update()
	clock.tick(FPS)




"""ENVIRONMENT CHANGES
1. recentre upon death
2. walls kill
3. overlap kill 
4. hard code prevention of direction swapping once length above 1. pIt's possible that because this causes a collision and a lost reward
the AI would have learnt this through trial and error but makes the game feel more realistic
5. added eyes to snake
6. allow manual quitting from PyGame console
7. flags for modes (good learning experience for a more professional feel N.B. access user_input via options.user_input)
8. obtain manhattan distance from snake head to target
9. prevent targets being created inside snake (not realistic) via a while loop in food creation function
10. user score and agent reward are different. included a negative reward for snake hitting itself (previously not there)
11. include a cumulative reward structure - necessary for RL goal of maximising long term future reward. this was tricky - needed to
	substantially alter the code to include a collision tracker to see what action env should take. couldn't use rewards for this.
12. Fixed some PyGame bugs to assist visualisations when AI mode enabled (previously freezing all the time)
13. added gridlines option to make move options more visible
14. realised q_table is not same size as map. in fact it continues growing as depends on current location and target location also.
	this increases memory requirement substantially. i think we can reduce this 
"""