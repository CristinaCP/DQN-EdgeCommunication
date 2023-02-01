from users import *
from operator import add
from combinatorics import *



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
v_R = 1
T_deadline = 50
alpha = 1e-3

class RoadSideUnit():
    def __init__(self, max_cache, total_cache_parts, num_content, num_cars, identifier, cache_options):
        #cache
        self.num_cars = num_cars
        self.num_content = num_content
        self.cache_capacity = max_cache
        self.min_cache_parts = 10
        self.total_content_parts = total_cache_parts
        self.cache_options = cache_options
        self.cache = self.update_cache()
        self.task_size = (self.l_c, self.l_c_w,  self.D_c) = (20, 2, 100) 
        self.request_count = np.array([0]*num_content)


        #ongoing computation
        self.ongoing_computation = np.array([0]*num_cars)

        #compute/CPU reserved
        self.min_compute = 10
        self.max_compute = 21
        self.total_compute = self.min_compute*num_cars
        self.cpu_reserved = np.array([0]*num_cars)
        self.identifier = identifier
        
    def reset(self):
        self.update_cache()
        self.update_computation(True)
        self.ongoing_computation = np.array([0]*num_cars)
        self.request_count = np.array([0]*num_content)

    def get_learning_state(self):
        state = np.array([self.identifier])
        state = np.append(state, self.cache)
        state = np.append(state, self.request_count)
        state = np.append(state, num_cars)
        return state.flatten()

    def step_small(self,request_list,channel_state, timestep):
        #update tasks
        penalty_list, transmit_list = self.update_task(request_list)
        # update comp
        self.update_computation()
        # calc cost, check termination
        cost = self.calculate_cost_small(penalty_list, transmit_list, channel_state, timestep)
        return cost, transmit_list

    def step_large(self,new_cache):
        #update cache 
        self.update_cache(self.cache_options[new_cache])
        #calc cost
        cost = self.calculate_cost_large()
        return cost

    def update_cache(self,cache_state= None):
        if cache_state == None:
          index = random.randrange(len(self.cache_options))
          return self.cache_options[index]
        else:
          self.cache = cache_state

    def update_computation(self,reset=False):
        if reset:
            self.cpu_reserved = np.array([random.choice(range(self.min_compute,self.max_compute+1)) for  c in range(self.num_cars)])
        else:
            num_CPU_states = 11
            trans_options = np.array([(1.0/num_CPU_states) for i in range(num_CPU_states)])
            self.cpu_reserved = np.random.choice(range(self.min_compute,self.max_compute),self.num_cars,p=trans_options)

    def update_task(self,request_list):
        penalty_list = []

        #print("request list:", request_list)
        #accept new tasks
        for (rsu, car_id, content, cycles) in request_list:
          if (rsu == self.identifier):
            self.request_count[content] += 1
            segments = self.cache[content]
            if segments == 0: 
                penalty_list.append((car_id,1))
            else:
                self.ongoing_computation[car_id] += cycles
        
        #process old ones
        transmit_list = []
        for car in range(len(self.ongoing_computation)):
            self.ongoing_computation[car] -= 1
            if self.ongoing_computation[car] <= 0: #compute task finished
                transmit_list.append((car,self.l_c))
                self.ongoing_computation[car] = 0

        return penalty_list, transmit_list


    def calculate_cost_small(self,penalty_list, transmit_list, channel_state, timestep):
        cost_per_car = [0]*self.num_cars
        #calculate cost of computation
        for car in range(len(self.ongoing_computation)):
            cost_per_car[car] += eta_RSU*D_c*e_RSU

        #cost transmit
        for (car,l_c) in transmit_list:
            cost_per_car[car] += delta_RSU*B*channel_state[car][self.identifier]*l_c 

        #cost memory/storage
        for car in range(self.num_cars):
            cost_per_car[car] += self.cpu_reserved[car]*eta_RSU

        #penalties
        for (car,pen) in penalty_list:
            cost_per_car[car] += pen*500

        return cost_per_car

    def calculate_cost_large(self):
        total_cost = 0

        #cost transmit
        cost_transmit = 0
        P_1 = lambda x: x/10 #TODO - probability that the vehicle received a segments
        for a in range(1,self.min_cache_parts):
            cost_transmit += P_1(a)*self.min_cache_parts*delta_RSU*B*v_R
            cost_transmit += P_1(a)*(a*delta_RSU*B*v_R - max(0, self.min_cache_parts -a)*(delta_BS*B_basestation*v_BS) )

        #cost computation/cache
        cost_compute = 0
        P_3 = lambda x : .3 #TODO - probability that vehicle offloads a tasks
        C_R = delta_RSU*B*v_R + eta_RSU*D_c*e_RSU
        C_BS = delta_BS*B_basestation*v_BS + eta_BS*D_c*e_BS
        sum_P_3 = 0
        for a in range(1,self.min_cache_parts):
            sum_P_3 += P_3(a)
        for c in range(self.num_content):
            for a in range(1,self.min_cache_parts):
                cost_compute += P_3(a)*(a*C_R - max(0, self.min_cache_parts-a)*C_BS)
            cost_compute += (1-sum_P_3)*self.min_cache_parts*C_R

        cost_compute = -cost_compute
        #cost mem/storage
        cost_storage = 0
        for c in range(self.num_content):
            #print("cost calc ", c, self.cache, self.cache[c])
            cost_storage += eta_RSU*(self.cache[c])

        total_cost = cost_compute + cost_storage + cost_transmit

        return total_cost

class NetworkEnvironment():
    def __init__(self,num_cars, num_rsu, num_content, lambda_movement,T_deadline, cache_options):
        #channel state
        self.num_cars = num_cars
        self.num_RSUs = num_rsu
        self.channel_state = self.update_channel(True)

        print(self.channel_state)
        #list of carsv and RSUs
        self.max_cache = 10
        self.num_content = num_content
        self.total_cache_parts = 20
        self.alpha = 1e-3
        self.cache_options = cache_options
        self.combin_item = Combinatorics(self.alpha, self.num_content)
        self.cars = [Vehicle(self.combin_item.choose_content(),num_rsu,lambda_movement,c) for c in range(num_cars)]
        self.RSUs = [RoadSideUnit(max_cache, self.total_cache_parts, num_content, num_cars, rsu, cache_options) for rsu in range(num_rsu)]

        #learning params 
        self.action_size = 2
        self.observation_size = (4) + num_rsu*2 + num_content

    #reset
    def reset(self):
        for car in self.cars:
            car.reset(self.combin_item.choose_content())
        for rsu in self.RSUs:
            rsu.reset()

        return self.get_learning_state_small(), self.get_learning_state_big()

    #step - cars
    def step_small(self, small_action_list, step):
        #take step
        cost_list,done_list,request_list = [],[],[]
        for car in self.cars:
            action = small_action_list[car.identifier]
            rsu = self.RSUs[car.location-1]
            available_compute = rsu.cpu_reserved[car.identifier]
            segments = rsu.cache[car.desired_content]
            request = car.take_step(action,available_compute, segments)
            if request != None:
              request_list.append(request)
            

        #update rsus
        total_transmit = np.array([0]*self.num_cars)
        cost_per_car = np.array([0]*self.num_cars)
        for rsu in self.RSUs:
            cost, transmit_list = rsu.step_small(request_list,self.channel_state, step)
            cost_per_car = list(map(add,cost_per_car,cost))
            for (car,amount) in transmit_list:
                total_transmit[car] += amount

        #calculate total cost per car
        for car in self.cars:
            car.receive_content(total_transmit[car.identifier])
            cost, done = car.calculate_cost(cost_per_car[car.identifier], step)
            done_list.append(done)
            cost_list.append(cost)

        return cost_list, done_list, request_list
    
    def update_channel(self, reset):
        (channel_min, channel_max) = (1,4)
        prob_change = 0.3 
        channel_state = np.array([[0 for rsu in range(self.num_RSUs)] for car in range(self.num_cars)])

        if reset:
            channel_state = np.array([[random.choice([channel_min,channel_max]) for rsu in range(self.num_RSUs)] for car in range(self.num_cars)])
        else:
            for rsu in range(self.num_RSUs):
                for car in range(self.num_cars):
                    current_val = self.channel_state[car][rsu]
                    trans_prob = random.uniform(0,1)
                    if trans_prob > prob_change: #transition
                        channel_state[car][rsu] = channel_min if current_val == channel_max else channel_max

        return channel_state

    #step - rsu, channel state
    def step_large(self, large_action):
        cost_list = []
        self.channel_state = self.update_channel(False)
        for rsu in self.RSUs:
            #update cache state in each rus
            cost = rsu.step_large(large_action[rsu.identifier])

            #estimate cost
            cost_list.append(cost)
            
        new_state_list = self.get_learning_state_big()

        return cost_list, new_state_list

    def print_state(self, filename, timestep, cost_list_small, cost_list_large=None ):
        #print state - for save_intermediate
        state = " ---- Timestep : " + str(timestep) + " ---- \r\n"
        total_cost = sum(cost_list_small)
        if cost_list_large != None:
          total_cost += sum(cost_list_large)
   
        #each car - location, desired content, request, received, decision, cost
        for car in self.cars:
            car_state = "---- Car " + str(car.identifier) + " at location " + str(car.location) + "\r\n"
            car_state += "Desired content : " + str(car.desired_content) + "\r\n"
            car_state += "Used RSUs : " + str(np.array(car.used_RSU).view()) + "\r\n"
            car_state += "Segments requested/received : " + str(car.content_requested) + "/" + str(car.content_received)  + "\r\n"
            car_state += " Cost: " +str( np.array(cost_list_small[car.identifier]).view()) + "\r\n"
            state += car_state

        #each rsu - identifier, cache, ongoing_computation, cost
        for rsu in self.RSUs:
            rsu_state = "---- RSU " + str(rsu.identifier) + "\r\n"
            rsu_state += "Cache : " + str(rsu.cache) + "\r\n"
            rsu_state += "Ongoing compute : " + str(np.array(rsu.ongoing_computation).view()) + "\r\n"
            if cost_list_large != None:
                rsu_state += "Cost : " + str(np.array(cost_list_large[rsu.identifier]).view()) + "\r\n"
            state += rsu_state

        #total cost
        state += "Total cost: " + str(total_cost) + "\r\n"

        #save to file
        f = open(filename, "a+")
        f.write(state)
        f.close()

    def get_learning_state_small(self):
        overall_state = []
        for car in self.cars:
            rsu = self.RSUs[car.location-1]
            cache_state = rsu.cache
            CPU_state = rsu.cpu_reserved[car.identifier]
            channel_state = self.channel_state[car.identifier][rsu.identifier]
            overall_state.append(car.get_learning_state(channel_state, cache_state, CPU_state))
        return overall_state

    def get_learning_state_big(self):
        overall_state = []
        for rsu in self.RSUs:
            overall_state.append(rsu.get_learning_state())
        return overall_state