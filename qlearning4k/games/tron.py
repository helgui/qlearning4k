from game import Game
import numpy as np

class Tron(Game):
	def __init__(self):
		self.points = []
		self.dirs = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
		
		for i in range(20):
			for j in range(30):
				self.points.append((i, j))
		
		self.points = np.array(self.points) 
				
		self.free = np.ones((4, 20, 30), dtype=np.bool)
		self.reset()
	
	@property 
	def name(self):
		return 'TronBattle' 
	
	@property
	def nb_actions():
		return 4

	def reset(self):
		self.players = np.random.randint(2, 5);
		self.active = np.ones(p, dtype=np.bool)
		self.pos = self.points[np.random.choice(20*30, size=self.p)]
		self.free.fill(True)

	def deactivate(self, idx):
		self.free[idx].fill(True)
		self.pos[idx] = (-1, -1)
		self.active[idx] = False

	def bad(self, point):
		if point[0] < 0 or point[1] < 0 or point[0] >= 20 or point[1] >= 30:
			return True
		return self.free[0:4, points[0], points[1]].all()

   	def turn(self, idx, action):
   		self.pos[idx] += self.dir[action]
   		if self.bad(self.pos):
   			self.deactivate(idx)
   		else:
   			self.free[idx][self.pos[idx][0]][self.pos[idx][1]] = False

    def is_over(self):
    	return (not self.active[0]) or (not self.active[1:].any())

    def is_won(self):
    	return self.is_over() and self.is_active(0)

    def get_score(self):
    	return 1.0 if self.active[0] else 0.0

    def get_possible_actions(self, idx = 0):
    	if (not self.active[idx])
    		return []
    	pa = []
    	for i in range(4):
    		self.pos[idx] += self.dirs[i]
    		if not self.bad(self.pos):
    			pa.append(i)
    		self.pos[idx] -= self.dirs[i]
    	if pa == []:
    		pa.append(np.random.randint(4))
		return pa

	def dist_field(self, idx):
		df = np.zeros((20, 30), dtype=np.float) 
		if not self.active[idx]:
			return df 
		q = Queue()
		q.put(self.pos)
		while not q.empty():
			p = q.get()
			for i in range(4):
				np = p + self.dirs[i] 
				if (not self.bad(np) and df[np[0]][np[1]] == 0.0):
					df[np[0]][np[1]] = df[p[0]][p[1]]
					
	def get_state(self):
		return np.array([self.dist_field(i) for i in range(self.players)]) 

   	def play(self, action):
   		self.turn(0, action)
   		for i in range(1, self.players):
   			if self.active[i]:
   				self.turn(i, )
   		
   		
   		
   		
