# DQN-EdgeCommunication

required packages
#best training example - https://github.com/tensorflow/agents/blob/master/tf_agents/colabs/1_dqn_tutorial.ipynb

# Note: If you haven't installed the following dependencies, run:
!apt-get install -y xvfb
!pip install -q pyyaml h5py
!pip install 'gym==0.10.11'
!pip install 'imageio==2.4.0'
!pip install PILLOW
!pip install 'pyglet==1.3.2'
!pip install pyvirtualdisplay
!pip install tf-agents-nightly
!pip install gast==0.2.2 

try:
  %%tensorflow_version 2.x
except:
  pass

!pip install keras-rl