um_content = 5
num_cars = 3
num_rsu = 5
max_cache = 10
lambda_movement = .2
T_deadline = 50

np.random.seed(0)

combin_item = Combinatorics(1e-3, num_content)
cache_options = combin_item.list_cache_combinations(num_content,max_cache)

train_env = NetworkEnvironment(num_cars, num_rsu, num_content, lambda_movement,T_deadline, cache_options)
eval_env = NetworkEnvironment(num_cars, num_rsu, num_content, lambda_movement,T_deadline, cache_options)

DQNAgent = Agent(train_env,False,cache_options)

#TRAIN THE AGENT
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")

intermediate_file = "interm-" + timestampStr + ".txt"
small_model_file = "small-" + timestampStr + ".txt"
large_model_file = "large-" + timestampStr + ".txt"

DQNAgent.train(intermediate_file, small_model_file, large_model_file)

#TEST
#test reload of data
#DQNAgent.test(eval_env, small_model_file, large_model_file)
