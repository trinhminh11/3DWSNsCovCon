data_num = 20     # Run how many time per 1 Dataset
dataset_num = 6  # Dataset
H = 2000	  # Region

n_step = 150		  # step of n
Rs_step = 15		  # step of Rs
Qmax_step = 2	  # step of Qmax

random_seed = 1

class Sensor:
	def __init__(self, v, R, Targets):
		self.v = v
		self.Targets = Targets

class Relay:
	def __init__(self, v):
		self.v = v
		self.prev = None

class Target:
	def __init__(self, v, q, Sensors):
		self.v = v
		self.q = q
		self.Sensors = Sensors

class Base:
	def __init__(self, v):
		self.v = v