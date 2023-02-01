import itertools


class Combinatorics():
  def __init__(self, alpha, C):
    self.combinations = []
    ro = 0
    for c in range(1,C+1):
      ro += 1/(c**alpha)
    self.C = C
    self.probability_content = [1/(ro*(c**alpha)) for c in range(1,C+1)]

  def choose_content(self):
    return np.random.choice(range(0,self.C),p=self.probability_content)

  # Function to find out all  
  # combinations of positive numbers  
  # that add upto given number. 
  # It uses findCombinationsUtil()  
  def findCombinations(self, n, num_content): 
    # array to store the combinations 
    # It can contain max n elements 
    arr = [0] * n; 

    # find all combinations 
    self.findCombinationsUtil(arr, 0, num_content, n, n); 

  def findCombinationsUtil(self, arr, index, num_content, num, reducedNum): 
    # Base condition 
    if (reducedNum < 0): 
      return; 
  
    # If combination is  
    # found, print it 
    if (reducedNum == 0): 
      no_trail = list(filter(lambda a: a != num, arr))
      if len(no_trail) == 0:
        no_trail = [10]
      if len(no_trail) <= num_content:
        padding = [0]*max((num_content-len(no_trail)),0)
        #print(no_trail + padding)
        self.combinations.append(no_trail + padding)
      return;
  
    # Find the previous number stored in arr[].  
    # It helps in maintaining increasing order 
    prev = 1 if(index == 0) else arr[index - 1]; 
  
    # note loop starts from previous  
    # number i.e. at array location 
    # index - 1 
    for k in range(prev, num + 1):  
      # next element of array is k 
      arr[index] = k; 

      # call recursively with 
      # reduced number 
      self.findCombinationsUtil(arr, index + 1, num_content, num, reducedNum - k); 

  def list_cache_combinations(self, num_content, max_cache):
      self.combinations = []
      self.findCombinations(max_cache, num_content)

      #print("Post combos:",len(self.combinations))
      all_permutations = []
      for i in range(len(self.combinations)):
          permutations_object = itertools.permutations(self.combinations[i])
          permutations_list =  list(dict.fromkeys(list(permutations_object)))
          #print(len(permutations_list),permutations_list)
          for j in range(len(permutations_list)):
              all_permutations.append(permutations_list[j])

      print(len(all_permutations)) #1001 for 10 objs and 5 content
      return all_permutations





