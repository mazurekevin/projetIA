import arcade
import os
import pickle
import time
import matplotlib.pyplot as plt

MAZE = """
############################################################
###@#####################@@@@@@@@@@@@@@@@@@@@@@#############
##@.@##################@@                      @############     
#@   @###############@@                         @###########
#@   @#############@@        @@@@@@@@@@@@@@@     @@#########
#@   @###########@@     @@@@@###############@      @########
#@    @########@@      @#####################@     @########
##@    @@#####@     @@@#####@@@@@@@@@@@@@####@      @#######
###@     @@@@@    @@#####@@@             @####@      @######
####@@           @####@@@                 @@###@     @######
######@@        @####@                      @###@    @######
########@@@@@@@@####@       @@@@@@@@@@@@     @@@@     @#####
###################@       @############@@            @#####
##################@       @###############@          @######
################@@       @#################@@@@@@@@@@#######
############@@@@        @###################################
###########@           @#################@@@@@@@@@@@@@@@@@##
##########@         @@@################@@               *@##
##########@       @@#################@@                 *@##
##########@      @##################@                   *@##
##########@      @#################@          @@@@@@@@@@@@##
##########@      @################@         @@##############
###########@      @@#############@        @@################
############@       @@@#######@@@       @@##################
#############@         @@###@@         @####################
##############@          @@@          @#####################
###############@                     @######################
################@@@                @@#######################
###################@@@@@@@@@@@@@@@@#########################
############################################################
"""

MAP_WALL = '@'
MAP_OUTSIDE = '#'
MAP_START = '.'
MAP_GOAL = '*'

REWARD_DEFAULT = -1

ACTION_UP = 'U'
ACTION_DOWN = 'D'
ACTION_LEFT = 'L'
ACTION_RIGHT = 'R'
ACTION_UP_RIGHT = 'UR'
ACTION_UP_LEFT = 'UL'
ACTION_DOWN_RIGHT = 'DR'
ACTION_DOWN_LEFT = 'DL'
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP_RIGHT, ACTION_UP_LEFT, ACTION_DOWN_LEFT,
           ACTION_DOWN_RIGHT]

ACTION_MOVE = {ACTION_UP: (-1, 0),
               ACTION_DOWN: (1, 0),
               ACTION_LEFT: (0, -1),
               ACTION_RIGHT: (0, 1),
               ACTION_UP_RIGHT: (1, 1),
               ACTION_UP_LEFT: (-1, 1),
               ACTION_DOWN_LEFT: (-1, -1),
               ACTION_DOWN_RIGHT: (1, -1)}

DIRECTION_UP = 'U'
DIRECTION_DOWN = 'D'
DIRECTION_LEFT = 'L'
DIRECTION_RIGHT = 'R'
DIRECTION_UP_RIGHT = 'UR'
DIRECTION_UP_LEFT = 'UL'
DIRECTION_DOWN_RIGHT = 'DR'
DIRECTION_DOWN_LEFT = 'DL'

DIRECTIONS = {DIRECTION_UP: 0,
              DIRECTION_UP_RIGHT: 45,
              DIRECTION_RIGHT: 90,
              DIRECTION_DOWN_RIGHT: 135,
              DIRECTION_DOWN: 180,
              DIRECTION_DOWN_LEFT: 225,
              DIRECTION_LEFT: 270,
              DIRECTION_UP_LEFT: 315}

SPRITE_SCALE = 0.2
SPRITE_CAR_SCALE = 0.04
SPRITE_FIELD_SCALE = 0.4
SPRITE_SIZE = round(128 * SPRITE_SCALE)

FILE_AGENT = 'agent.al1'
DIR_SPRITES = os.getcwd() + '/sprites/'


class Environment:
    def __init__(self, str_map):
        row = 0
        col = 0
        self.__states = {}
        str_map = str_map.strip()
        for line in str_map.strip().split('\n'):
            for item in line:
                self.__states[row, col] = item
                if item == MAP_GOAL:
                    self.__goal = (row, col)
                elif item == MAP_START:
                    self.__start = (row, col)
                col += 1
            row += 1
            col = 0

        self.__rows = row
        self.__cols = len(line)

        self.__reward_goal = len(self.__states)
        self.__reward_wall = -2 * self.__reward_goal

    def do(self, state, action):
        #time.sleep(0.1)
        move = ACTION_MOVE[action]
        new_state = (state[0] + move[0], state[1] + move[1])

        if new_state not in self.__states \
                or self.__states[new_state] in [MAP_OUTSIDE, MAP_WALL, MAP_START]:
            reward = self.__reward_wall
        else:
            state = new_state
            if new_state == self.__goal:
                reward = self.__reward_goal
            else:
                reward = REWARD_DEFAULT

        return state, reward

    @property
    def states(self):
        return list(self.__states.keys())

    @property
    def start(self):
        return self.__start

    @property
    def goal(self):
        return self.__goal

    @property
    def height(self):
        return self.__rows

    @property
    def width(self):
        return self.__cols

    def is_wall(self, state):
        return self.__states[state] == MAP_WALL

    def is_outside(self, state):
        return self.__states[state] == MAP_OUTSIDE

    def is_goal(self, state):
        return self.__states[state] == MAP_GOAL


class Agent:
    def __init__(self, env, alpha=1, gamma=0.5):
        self.__qtable = {}
        for state in env.states:
            self.__qtable[state] = {}
            for action in ACTIONS:
                self.__qtable[state][action] = 0.0

        self.__env = env
        self.__alpha = alpha
        self.__gamma = gamma
        self.__action = ACTION_UP
        # self.__angle = 0
        self.__history = []
        self.reset(False)

    def reset(self, store_history=True):
        if store_history:
            self.__history.append(self.__score)
        self.__state = env.start
        self.__score = 0

    def best_action(self):
        q = self.__qtable[self.__state]
        return max(q, key=q.get)

    def step(self):
        action = self.best_action()
        self.__action = action
        # self.rotate_sprite(self.__state, action)
        print(self.__state, action)
        state, reward = self.__env.do(self.__state, action)

        maxQ = max(self.__qtable[state].values())
        delta = self.__alpha * (reward + self.__gamma * maxQ - self.__qtable[self.__state][action])
        self.__qtable[self.__state][action] += delta

        self.__state = state
        self.__score += reward
        return action, reward

    # def rotate_sprite(self, point, action):
    #     if action == ACTION_UP:
    #         self.rotate_around_point(point, 0)
    #     elif action == ACTION_LEFT:
    #         self.rotate_around_point(point, 270)
    #     elif action == ACTION_RIGHT:
    #         self.rotate_around_point(point, 90)
    #     elif action == ACTION_DOWN:
    #         self.rotate_around_point(point, 180)
    #     elif action == ACTION_UP_LEFT:
    #         self.rotate_around_point(point, 315)
    #     elif action == ACTION_UP_RIGHT:
    #         self.rotate_around_point(point, 45)
    #     elif action == ACTION_DOWN_LEFT:
    #         self.rotate_around_point(point, 225)
    #     elif action == ACTION_DOWN_RIGHT:
    #         self.rotate_around_point(point, 135)

    # def rotate_around_point(self, point: arcade.Point, degrees: float):
    #     # Make the sprite turn as its position is moved
    #     self.__angle = degrees
    #     print(point, self.__angle)
    #
    #     center_x, center_y = point
    #     # Move the sprite along a circle centered around the passed point
    #     arcade.rotate_point(
    #         center_x, center_y,
    #         point[0], point[1], degrees)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.__qtable, self.__history = pickle.load(file)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.__qtable, self.__history), file)

    @property
    def action(self):
        return self.__action

    @property
    def state(self):
        return self.__state

    @property
    def score(self):
        return self.__score

    @property
    def environment(self):
        return self.__env

    @property
    def history(self):
        return self.__history

    def __repr__(self):
        res = f'Agent {agent.state}\n'
        res += str(self.__qtable)
        return res


class MazeWindow(arcade.Window):
    def __init__(self, agent):
        super().__init__(SPRITE_SIZE * agent.environment.width,
                         SPRITE_SIZE * agent.environment.height, "Micromachine")

        self.__agent = agent
        self.__iteration = 0

    def setup(self):
        self.__walls = arcade.SpriteList()
        for state in filter(self.__agent.environment.is_wall,
                            self.__agent.environment.states):
            sprite = arcade.Sprite(':resources:images/topdown_tanks/tileSand1.png', SPRITE_FIELD_SCALE)
            sprite.center_x, sprite.center_y = self.state_to_xy(state)
            self.__walls.append(sprite)

        self.__outsides = arcade.SpriteList()
        for state in filter(self.__agent.environment.is_outside,
                            self.__agent.environment.states):
            sprite = arcade.Sprite(':resources:images/topdown_tanks/tileGrass2.png', SPRITE_FIELD_SCALE)
            sprite.center_x, sprite.center_y = self.state_to_xy(state)
            self.__outsides.append(sprite)

        self.__player_up = arcade.Sprite(os.path.join(DIR_SPRITES, 'sprite_car_yellow_up.png'), SPRITE_CAR_SCALE)
        self.__player_down = arcade.Sprite(os.path.join(DIR_SPRITES, 'sprite_car_yellow_down.png'), SPRITE_CAR_SCALE)
        self.__player_left = arcade.Sprite(os.path.join(DIR_SPRITES, 'sprite_car_yellow_left.png'), SPRITE_CAR_SCALE)
        self.__player_right = arcade.Sprite(os.path.join(DIR_SPRITES, 'sprite_car_yellow_right.png'), SPRITE_CAR_SCALE)
        self.__player_up_right = arcade.Sprite(os.path.join(DIR_SPRITES, 'sprite_car_yellow_up_right.png'),
                                               SPRITE_CAR_SCALE)
        self.__player_up_left = arcade.Sprite(os.path.join(DIR_SPRITES, 'sprite_car_yellow_up_left.png'),
                                              SPRITE_CAR_SCALE)
        self.__player_down_right = arcade.Sprite(os.path.join(DIR_SPRITES, 'sprite_car_yellow_down_right.png'),
                                                 SPRITE_CAR_SCALE)
        self.__player_down_left = arcade.Sprite(os.path.join(DIR_SPRITES, 'sprite_car_yellow_down_left.png'),
                                                SPRITE_CAR_SCALE)

        self.__goal = arcade.SpriteList()
        for state in filter(self.__agent.environment.is_goal,
                            self.__agent.environment.states):
            sprite = arcade.Sprite(os.path.join(DIR_SPRITES, 'sprite_goal.png'), 0.04)
            sprite.center_x, sprite.center_y = self.state_to_xy(state)
            self.__goal.append(sprite)

        self.__sound = arcade.Sound(':resources:sounds/rockHit2.wav')

    def state_to_xy(self, state):
        return (state[1] + 0.5) * SPRITE_SIZE, \
               (self.__agent.environment.height - state[0] - 0.5) * SPRITE_SIZE

    def on_draw(self):
        arcade.start_render()
        self.__walls.draw()
        self.__outsides.draw()
        self.__goal.draw()

        if self.__agent.action == ACTION_UP:
            self.__player_up.draw()
        elif self.__agent.action == ACTION_LEFT:
            self.__player_left.draw()
        elif self.__agent.action == ACTION_RIGHT:
            self.__player_right.draw()
        elif self.__agent.action == ACTION_DOWN:
            self.__player_down.draw()
        elif self.__agent.action == ACTION_UP_RIGHT:
            self.__player_up_right.draw()
        elif self.__agent.action == ACTION_UP_LEFT:
            self.__player_up_left.draw()
        elif self.__agent.action == ACTION_DOWN_LEFT:
            self.__player_down_left.draw()
        elif self.__agent.action == ACTION_DOWN_RIGHT:
            self.__player_down_right.draw()

        arcade.draw_text(f'#{self.__iteration} Score: {self.__agent.score}', 10, 10,
                         arcade.csscolor.WHITE, 20)

    def on_update(self, delta_time):
        if self.__agent.state != self.__agent.environment.goal:
            self.__agent.step()
            self.__player_up.center_x, self.__player_up.center_y = self.state_to_xy(self.__agent.state)
            self.__player_left.center_x, self.__player_left.center_y = self.state_to_xy(self.__agent.state)
            self.__player_right.center_x, self.__player_right.center_y = self.state_to_xy(self.__agent.state)
            self.__player_down.center_x, self.__player_down.center_y = self.state_to_xy(self.__agent.state)
            self.__player_down_right.center_x, self.__player_down_right.center_y = self.state_to_xy(self.__agent.state)
            self.__player_down_left.center_x, self.__player_down_left.center_y = self.state_to_xy(self.__agent.state)
            self.__player_up_left.center_x, self.__player_up_left.center_y = self.state_to_xy(self.__agent.state)
            self.__player_up_right.center_x, self.__player_up_right.center_y = self.state_to_xy(self.__agent.state)
        else:
            self.__agent.reset()
            self.__iteration += 1
            # self.__sound.play()


if __name__ == "__main__":
    env = Environment(MAZE)
    print(f'Number of states {len(env.states) * 4}')
    agent = Agent(env)
    agent2 = Agent(env)
    agents = [agent,agent2]

    if os.path.exists(FILE_AGENT):
        agent.load(FILE_AGENT)
        plt.plot(agent.history)
        plt.show()

    window = MazeWindow(agent)
    window.setup()
    #window.set_visible(False)
    arcade.run()

    agent.save(FILE_AGENT)
