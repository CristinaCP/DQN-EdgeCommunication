class Vehicle():
    epsilon_c = 2 #units/MB
    B = 1 # MHz
    B_basestation = 4*B  #MHz 
    delta_BS = 20 #units/MHz
    delta_RSU = 2 #units/MHz
    eta_BS = 100  #units/J
    eta_RSU = 10 #units/J
    e_BS = e_RSU = 1 #W/GHz
    l_c = 20 #bits, size of task 
    D_c = 100 #Mcycles, CPU cycles to complete task
    v_BS = 2
    T_deadline = 50

    def __init__(self,content,num_RSU,lambda_movement,identifier):
        self.desired_content = content
        self.num_RSU = num_RSU
        self.location = random.choice(range(num_RSU))
        self.content_received = 0
        self.content_requested = 0
        self.used_RSU = np.array([0]*num_RSU)
        self.mobility_trans = self.create_movement_trans()
        self.min_cache_parts = 10
        self.penalty = 0
        self.lambda_movement = lambda_movement
        self.identifier = identifier

    def reset(self,content):
        self.desired_content = content
        self.content_received = 0
        self.content_requested = 0
        self.used_RSU = np.array([0]*self.num_RSU)
        self.penalty = 0

    def get_learning_state(self, channel_state, cache_state, CPU_state):
        state = np.array([self.identifier,self.location, self.desired_content, self.lambda_movement])
        state = np.append(state, cache_state)
        state = np.append(state, CPU_state )
        state = np.append(state, channel_state)
        state = np.append(state, self.used_RSU)
        #other = CPU_state + [channel_state] + self.used_RSU.flatten()
        return state.flatten()


    def take_step(self,action, available_compute, segments):
        #take action
        next_location = self.update_location()
        next_task_state, penalty, request = self.update_task_state(action, available_compute, segments)

        #update object attributes
        self.location = next_location
        self.content_requested = next_task_state
        self.penalty = penalty
        
        return request

    def update_task_state(self,action, available_compute, segments):
        penalty = 0
        request = None
        remaining_segments = 0
        if action:
            if self.used_RSU[self.location-1]:
                penalty += 1 
            else:
                remaining_segments = min(self.min_cache_parts - self.content_requested, segments)
                if remaining_segments > 0:
                    number_of_cycles = (remaining_segments*self.D_c)/available_compute
                    request = (self.location,self.identifier,self.desired_content,number_of_cycles)
                    self.used_RSU[self.location-1] = 1
                    self.content_requested += remaining_segments
        else:
            if segments != 0 and not(self.used_RSU[self.location-1]):
                penalty = 1

        #print("Car : ", self.identifier, "request: ", request)
        return remaining_segments, penalty, request
    
    def update_location(self):
        trans_options = self.mobility_trans[self.location] #array of potential transition from current to next
        next_loc = np.random.choice(range(self.num_RSU+1), p=trans_options)
        return next_loc

    def receive_content(self,value):
        self.content_received += value

    def create_movement_trans(self):
        forward = .6; 
        back = 1-forward; 
        K = self.num_RSU

        total_trans = []
        for start_loc in range(K+1):
          trans = [0 for k in range(K+1)]
          if start_loc == 0:
              trans[1] = back 
              trans[0] = forward 
          elif start_loc == K:
              trans[K-1] = back
              trans[K] = forward 
          else:
              trans[start_loc-1] = back
              trans[start_loc+1] = forward
          total_trans.append(trans)
            
        return total_trans
    
    def check_terminated(self, cost):
        if cost <= 0:
            return True
        if self.content_received == self.content_requested == self.min_cache_parts:
            return True
        
        return False

    def calculate_cost(self, cost, steps):
        ungotten_sections = self.min_cache_parts - self.content_received
        if (steps % T_deadline) == 0:
            #get unfetched components from the base station
            cost += delta_RSU*B_basestation*v_BS*(ungotten_sections*l_c) + eta_BS*(D_c*ungotten_sections)*e_BS
            self.used_RSU = np.array([0]*self.num_RSU)
        done = self.check_terminated(cost)

        return -cost, done
