import pygame
import math
import random
import gym
from gym import spaces
import numpy as np

class Player():
    def __init__(self, x, y):
        self.x = 370
        self.y = 480
        self.rect = pygame.Rect(self.x, self.y, 32, 32)
        self.width = 32
        self.height = 32
        self.velX = 0
        self.velY = 0
        #self.image = pygame.image.load('player.png').convert() # This is not needed for training
        self.speed = 5
    def step(self, action):
        if action == 0:  # Move left
            self.x -= self.speed
        elif action == 1:  # Move right
            self.x += self.speed
        elif action == 2:  # Move up
            self.y -= self.speed
        elif action == 3:  # Move down
            self.y += self.speed
        elif action == 4:  # Move up and left
            self.x -= self.speed
            self.y -= self.speed
        elif action == 5:  # Move up and right
            self.x += self.speed
            self.y -= self.speed
        elif action == 6:  # Move down and left
            self.x -= self.speed
            self.y += self.speed
        elif action == 7:  # Move down and right
            self.x += self.speed
            self.y += self.speed
        elif action == 8:  # Do nothing
            pass
        self.rect.x = self.x
        self.rect.y = self.y
        if self.x < 0:
            self.x = 0
        elif self.x > 600 - self.width:
            self.x = 600 - self.width
        if self.y < 0:
            self.y = 0
        elif self.y > 600 - self.height:
            self.y = 600 - self.height
    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))
        pygame.draw.rect(screen, (0, 255, 0), self.rect, 2)

class Bullet():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, 8, 8)
        self.width = 8
        self.height = 8
        #self.image = pygame.image.load('bullet.png').convert() # This is not needed for training
        target_x = random.randint(50, 550)
        target_y = random.randint(50, 550)
        self.angle = math.atan2(target_y - self.y, target_x - self.x)
        self.speed = 5
        self.lifetime = 0
    def handle_events(self, event):
        pass
    def update(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.lifetime += 1
        self.rect.x = self.x
        self.rect.y = self.y
    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))
        pygame.draw.rect(screen, (255, 0, 0), self.rect, 2)

class DodgeBulletsEnv(gym.Env):
    def __init__(self):
        super(DodgeBulletsEnv, self).__init__()

        # Define the action space and observation space
        self.action_space = spaces.Discrete(9)  # Five discrete actions: 0, 1, 2, 3, 5
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 6), dtype=np.uint8)

        # Initialise the game variables
        self.difficulty = 10.0
        self.spawn_rate = self.difficulty

        # Initialise pygame
        pygame.init()

        # Initialise the player, bullets and clock
        self.player = Player(370, 480)
        self.bullets = []
        self.clock = pygame.time.Clock()

        # Create the screen
        self.screen_width, self.screen_height = 600, 600
        #self.screen = pygame.display.set_mode((self.screen_width, self.screen_height)) # This creates the screen but is not needed for training

    def _get_state(self, action=0):
        state = [0] * 14  # Initialize the state array with zeros

        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.width, self.player.height)
        detection_range = 64

        # Calculate distances and angles for the closest three bullets
        closest_bullets = []
        closest_distances = []
        closest_angles = []

        # Check for bullets within the detection range around the player
        for bullet in self.bullets:
            bullet_rect = pygame.Rect(bullet.x, bullet.y, bullet.width, bullet.height)
            if player_rect.colliderect(bullet_rect.inflate(detection_range, detection_range)):
                # Determine the relative direction of the bullet from the center of the player
                dx = bullet.x - (self.player.x + self.player.width / 2)
                dy = bullet.y - (self.player.y + self.player.height / 2)

                distance = math.sqrt(dx ** 2 + dy ** 2)
                angle = math.atan2(dy, dx)

                # Update the closest bullets list
                if len(closest_bullets) < 3:
                    closest_bullets.append(bullet)
                    closest_distances.append(distance)
                    closest_angles.append(angle)
                else:
                    # Replace the farthest bullet if the current one is closer
                    farthest_index = closest_distances.index(max(closest_distances))
                    if distance < closest_distances[farthest_index]:
                        closest_bullets[farthest_index] = bullet
                        closest_distances[farthest_index] = distance
                        closest_angles[farthest_index] = angle

        # Set the state values for the closest bullets
        for i, bullet in enumerate(closest_bullets):
            state[5 + i * 3] = closest_distances[i]
            state[6 + i * 3] = closest_angles[i]
            state[7 + i * 3] = bullet.angle

        state[4] = action

        # Check player's position on the edge
        if self.player.y == 0:
            state[0] = 1
        if self.player.y == self.screen_height - self.player.height:
            state[1] = 1
        if self.player.x == 0:
            state[2] = 1
        if self.player.x == self.screen_width - self.player.width:
            state[3] = 1

        return state
    
    def _calculate_reward(self, action, collision):
        if collision:
            reward = -100  # Negative reward for collision
        elif action == 8:
            reward = 2  # Greater reward for not moving, this incentivises the agent to not move when possible
        else:
            reward = 1 # Small positive reward for not colliding and moving

        return reward

    def step(self, action):
        # Execute the action
        # Update the game state based on the action
        collision = False
        self.player.step(action)

        # Timer for bullet spawn, when spawning it will pick a random side and spawn a bullet there, bullets move approximately towards the center of the screen
        # Spawn rate resets to the difficulty value which by default is 10 (one bullet every 10 frames) and decreases by 0.01 every spawn
        if self.spawn_rate > 0:
            self.spawn_rate -= 1
        else:
            side = random.randint(1, 4)  # 1: top, 2: right, 3: bottom, 4: left
            if side == 1:  # Top
                x = random.randint(0, 600)
                y = random.randint(-50, -10)
            elif side == 2:  # Right
                x = random.randint(600 + 10, 600 + 50)
                y = random.randint(0, 600)
            elif side == 3:  # Bottom
                x = random.randint(0, 600)
                y = random.randint(600 + 10, 600 + 50)
            else:  # Left
                x = random.randint(-50, -10)
                y = random.randint(0, 600)
            self.bullets.append(Bullet(x, y))
            self.difficulty -= 0.01
            self.spawn_rate = self.difficulty

        # Updating the bullets, if the bullet is older than 10000 frames it will be removed from the list
        for bullet in self.bullets:
            bullet.update()
            if pygame.Rect.colliderect(self.player.rect, bullet.rect):
                collision = True
            if bullet.lifetime > 10000:
                self.bullets.remove(bullet)

        # Get the current state
        state = self._get_state(action)

        # Calculate the reward based on the game state
        reward = self._calculate_reward(action, collision)

        # Check if the game is over
        done = collision

        # Return the next state, reward, done flag, and any additional information
        return state, reward, done, {}

    def render(self, mode="human"):
        # Render the game screen
        # Commented out here since we do not want to render the game window when training
        """
        self.screen.fill((0, 0, 0))
        self.player.render(self.screen)
        for bullet in self.bullets:
            bullet.render(self.screen)
        
        pygame.display.update()
        """
        return
    
    def reset(self):
        # Reset the game state to the initial state
        self.player.x = 370
        self.player.y = 480

        self.bullets.clear()

        self.difficulty = 10.0
        self.spawn_rate = self.difficulty

        # Return the initial state
        return self._get_state()

# Building the model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.callbacks import Callback

env = DodgeBulletsEnv() 

states = env.observation_space.shape
actions = env.action_space.n

# Function to build the model
def build_model(states, actions):
    model = Sequential() # Neural network
    
    model.add(Flatten(input_shape=(1, 14))) # Input layer
    model.add(Dense(16, activation='relu')) # Hidden layer
    model.add(Dense(actions, activation='linear')) # Output layer
    return model

model = build_model(states, actions)
model.summary()

# Building the agent
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                    nb_actions=actions, nb_steps_warmup=1000, target_model_update=1e-2)
    return dqn

# Training the agent
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=1)

# Save the model
dqn.save_weights('first_model.h5f', overwrite=True)