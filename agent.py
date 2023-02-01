from networkenvironment import *
from combinatorics import * 
from operator import add
import numpy as np

class Agent():
    def __init__(self,environment,testing,cache_options):
        self.env = environment
        self.action_size = environment.action_size
        self.observation_size = environment.observation_size
        
        self.exp_replay_small = []
        self.exp_replay_large = []
        self.max_mem = 10000 
        self.cache_options = cache_options

        #learning parameters 
        self.gamma = .6 #discount factor
        self.epsilon = .4 #exploration rate
        self.alpha = 1e-3 #learning rate
        self.learning_rate =  1e-3
        self.batch_size = 64
        self.num_episodes = 100
        self.steps_per_episode = 100
        self.discount_factor = .6
        self.T_deadline = 50
        self.T_large = 10

        self.small_step_network = self._build_model_small()
        self.large_step_network = self._build_model_large()

        self.testing = testing
        self.evaluating = False
    
    def print_info(self,steps):
        print("Agent QNetwork Info")
        print("Size of Inputs (action, obs): ", self.action_size, self.observation_size)
        print("Size of Environments (cars, rsu): ", self.env.num_cars, self.env.num_RSUs)
        if(len(self.exp_replay_small) > 0):
            print("Experience replay : ", len(self.exp_replay_small), self.exp_replay_small[0])
        if(len(self.exp_replay_large) > 0):
            print("Experience replay : ", len(self.exp_replay_large), self.exp_replay_large[0])

        print(self.small_step_network.summary())
        print(self.large_step_network.summary())
        print("Step: ",steps)

    #### TESTING FUNCTIONS ####
    def reload(self, filepath_small, filepath_large):
        self.small_step_network.load_weights(filepath_small)
        self.large_step_network.load_weights(filepath_large)
        print("Network Summaries: \r\n" + self.small_step_network.summary() + "\r\n " + self.large_step_network.summary())

    def evaluate(self, test_objects, test_outcomes):
        loss, acc = self.small_step_network.evaluate(test_objects, test_outcomes, verbose=2)
        print("Restored small model, accuracy: {:5.2f}%".format(100*acc))
        loss, acc = self.large_step_network.evaluate(test_objects, test_outcomes, verbose=2)
        print("Restored large model, accuracy: {:5.2f}%".format(100*acc))

    def test(self,eval_env,filepath_small, filepath_large):
        self.evaluating = True

        self.reload(filepath_small, filepath_large)
        self.print_info(0)

        total_reward_list = [0]*self.env.num_cars
        for e in range(self.num_episodes):
            small_state_list, big_state_list = eval_env.reset()
            for timestep in range(self.steps_per_episode):
                #small update
                small_state_list, reward_list, done_list = self.act_small(e, timestep, small_state_list)
                total_reward_list = list(map(add,total_reward_list,reward_list))

                #large update
                if timestep > 0 and timestep % self.T_large == 0:
                    big_state_list, estimate_list = self.act_large(e, timestep, big_state_list)

            avg = np.true_divide(np.array(total_reward_list), self.steps_per_episode)
            avg_rewards.append(avg)

            print(e, np.mean(avg), avg.view())
        self.print_info(100)


    #### LEARNING FUNCTIONS ####
    def _build_model_small(self):
        model = tf.keras.models.Sequential() 
        model.add(tf.keras.layers.Flatten(input_shape = (1,)))

        model.add(tf.keras.layers.Dense(300, activation="relu") )

        '''
        model.add(tf.keras.layers.Dense(200, activation="relu")) 
        model.add(tf.keras.layers.Dense(128, activation="relu")) 
        model.add(tf.keras.layers.Dense(64, activation="relu")) 
        model.add(tf.keras.layers.Dense(16, activation="relu")) 
        '''
        model.add(tf.keras.layers.Dense(128, activation="relu")) 
        model.add(tf.keras.layers.Dense(24, activation="relu")) 
        model.add(tf.keras.layers.Dense(self.action_size, activation = "linear"))  #a q value for each possible action
        optimizer1 = tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer1)
        return model
    
    def _build_model_large(self):
        build_model_size = len(self.cache_options)
        model = tf.keras.models.Sequential() 
        model.add(tf.keras.layers.Flatten(input_shape = (1,)))
        model.add(tf.keras.layers.Dense(300, activation="relu") )
        model.add(tf.keras.layers.Dense(200, activation="relu")) 
        model.add(tf.keras.layers.Dense(128, activation="relu")) 
        model.add(tf.keras.layers.Dense(64, activation="relu")) 
        model.add(tf.keras.layers.Dense(16, activation="relu")) 

        model.add(tf.keras.layers.Dense(build_model_size, activation = "linear"))  #a q value for each possible action
        optimizer1 = tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer1)
        return model

    def act_small(self, episode, step, state_list):
        small_action_list = []
        for car in self.env.cars:
            #choose action from DQN
            #print("State list : ", state_list, "ID: ", car.identifier)
            current_state = state_list[car.identifier]
            action = self.choose_action(current_state)
            small_action_list.append(action)

        #take action, update state lists
        cost_list, done_list, _ = self.env.step_small(small_action_list, step)

        #update experience replay
        new_state_list = self.env.get_learning_state_small()
        if not self.evaluating:
            for car in self.env.cars: 
                i = car.identifier
                self.exp_replay_small.append((state_list[i], small_action_list[i], cost_list[i], new_state_list[i], done_list[i]))
            if len(self.exp_replay_small) > self.max_mem:
                for i in range(self.env.num_cars):
                    self.exp_replay_small.pop(0)
        
            if episode > 10:
                self.update_network(True)
                if episode % 3 == 0:
                  self.update_network(False)
                

        return new_state_list, cost_list, done_list

    def choose_action(self,state):
        #testing acts randomly always, exploration phase chooses random action
        if not self.evaluating and (self.testing or (np.random.rand() < self.epsilon)):
            return random.randrange(self.action_size)

        #exploitation - choose action with highest q val
        #print("Choose action state: ", state)
        q_values = self.small_step_network.predict(np.array(state) )
        return np.argmax(q_values[0])


    def act_large(self, episode, step, state_list):
        large_action_list = []
        #choose action from DQN
        for rsu in self.env.RSUs:
            action = self.choose_cache(state_list[rsu.identifier])
            large_action_list.append(action)

        #take action, update state lists
        #print("Large Action List", large_action_list)
        cost_list, new_state_list = self.env.step_large(large_action_list)

        #update experience replay
        if not self.evaluating:
            for rsu in self.env.RSUs: 
                i = rsu.identifier
                self.exp_replay_large.append((state_list[i], large_action_list[i], cost_list[i], new_state_list[i]))
            if len(self.exp_replay_large) > self.max_mem:
                for i in range(self.env.num_rsu):
                    self.exp_replay_large.pop(0)
            
            if episode > 10:
                self.update_network(False)

        return new_state_list, cost_list

    def choose_cache(self,state):
        if not self.evaluating and (self.testing or (np.random.rand() < self.epsilon)):
            index = random.randrange(len(self.cache_options))
            return index
        else:
            q_values = self.large_step_network.predict(np.array(state))
            return np.argmax(q_values[0])

    def update_network(self, isSmall):
        if isSmall:
            model = self.small_step_network
            exp_replay = self.exp_replay_small

            if self.testing or (len(exp_replay) < self.batch_size):
                return
            batch = random.sample(exp_replay,self.batch_size)
            for state, action, reward,next_state,done in batch:
                q_update = self.alpha * (reward + self.gamma * np.max(model.predict(np.array(next_state))[0])) #reward - self.alpha * (reward + self.gamma * np.max(self.target_network.predict(next_state)[0])) 
                q_values = model.predict(np.array(state))
                q_values[0][action] = q_update
                model.fit(state,q_values, verbose=0)

        else:
            model = self.large_step_network
            exp_replay = self.exp_replay_large

            
            if self.testing or (len(exp_replay) < self.batch_size):
                return
            batch = random.sample(exp_replay,self.batch_size)
            for state, action, reward,next_state in batch:
                q_update = self.alpha * (reward + self.gamma * np.max(model.predict(np.array(next_state))[0])) #reward - self.alpha * (reward + self.gamma * np.max(self.target_network.predict(next_state)[0])) 
                q_values = model.predict(np.array(state))
                q_values[0][action] = q_update
                model.fit(state,q_values, verbose=0)


        

    def train(self,init_file,small_file, large_file):
        self.evaluating = False
        avg_rewards = []
        self.print_info(0)
        for e in range(self.num_episodes):
            small_state_list, big_state_list = self.env.reset()
            total_reward_list = [0]*self.env.num_cars
            for timestep in range(self.steps_per_episode):
                #small update
                small_state_list, reward_list, done_list = self.act_small(e, timestep, small_state_list)
                
                total_reward_list = list(map(add,total_reward_list,reward_list))
                #large update
                if timestep > 0 and timestep % self.T_large == 0:
                    big_state_list, estimate_list = self.act_large(e, timestep, big_state_list)
                    self.env.print_state(init_file, timestep, total_reward_list, estimate_list )
                else:
                    self.env.print_state(init_file, timestep, total_reward_list, None)
                

            avg = np.true_divide(np.array(total_reward_list), self.steps_per_episode)
            avg_rewards.append(avg)

            print(e, np.mean(avg), avg.view())

            self.small_step_network.save(small_file)
            self.large_step_network.save(large_file)

            #reduce exploration rate over time 
            if e % 5 == 0:
                self.epsilon = max(0, self.epsilon - .05)
        self.print_info(100)

