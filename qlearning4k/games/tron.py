from .game import Game
import numpy as np
from queue import Queue

class Tron(Game):
	def __init__(self, model):
		self.model = model
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
	def nb_actions(self):
		return 4

	def reset(self):
		self.players = np.random.randint(2, 5);
		self.active = np.zeros(4, dtype=np.bool)
		for i in range(self.players):
			self.active[i] = True
		self.pos = self.points[np.random.choice(20*30, size=self.players)]
		self.free.fill(True)
		for i in range(self.players):
			self.free[i][self.pos[i][0]][self.pos[i][1]] = False

	def deactivate(self, idx):
		self.free[idx].fill(True)
		self.active[idx] = False

	def is_free(self, point):
		return self.free[0:4, point[0], point[1]].all()
	
	def bad(self, point):
		if point[0] < 0 or point[1] < 0 or point[0] >= 20 or point[1] >= 30:
			return True
		return not self.is_free(point)

	def turn(self, idx, action):
		self.pos[idx] += self.dirs[action]
		if self.bad(self.pos[idx]):
			self.deactivate(idx)
		else:
			self.free[idx][self.pos[idx][0]][self.pos[idx][1]] = False

	def play(self, action):
		self.turn(0, action)
		for i in range(1, self.players):
			if self.active[i]:
				self.turn(i, self.ai(i))

	def is_over(self):
		return (not self.active[0]) or (not self.active[1:].any())

	def is_won(self):
		return self.is_over() and self.active[0]

	def get_score(self):
		return np.count_nonzero(np.logical_not(self.free[0]))	

	def get_possible_actions(self, idx = 0):
		if (not self.active[idx]):
			return []
		pa = []
		for i in range(4):
			self.pos[idx] += self.dirs[i]
			if not self.bad(self.pos[idx]):
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
		q.put(self.pos[idx])
		while not q.empty():
			p = q.get()
			for i in range(4):
				npos = p + self.dirs[i] 
				if (not self.bad(npos) and df[npos[0]][npos[1]] == 0.0):
					df[npos[0]][npos[1]] = df[p[0]][p[1]]
		return df
					
	def get_state(self, idx=0):
		order = list(range(4))
		if idx > 0:
			order[0], order[idx] = order[idx], order[0]
		return np.array([self.dist_field(i) for i in order])

	def get_frame(self):
		return self.get_state(0)

	def ai(self, idx):
		state = np.array([[self.get_state(idx)]])
		q = self.model.predict(state)[0]
		possible_actions = self.get_possible_actions(idx)
		q = [q[i] for i in possible_actions]
		return possible_actions[np.argmax(q)]

	def update_ai_model(self, model):
		self.model.set_weights(model.get_weights())

	def draw(self):
		out = np.zeros((20, 30), dtype=np.float)
		for i in range(20):
			for j in range(30):
				if not self.free[0][i][j]:
					out[i][j] = 0.5
				elif self.is_free((i, j)):
					out[i][j] = 0
				else:
					out[i][j] = 1.0
		return out

	def draw_img(self):
		zoom = 30
		out = np.zeros((20*zoom, 30*zoom, 3), dtype=np.uint8)
		for i in range(20):
			for j in range(30):
				if not self.free[0][i][j]:
					out[i*zoom:(i+1)*zoom, j*zoom:(j+1)*zoom] = (255, 0, 0)
				elif self.is_free((i, j)):
					out[i*zoom:(i+1)*zoom, j*zoom:(j+1)*zoom] = (255, 255, 255)
				else:
					out[i*zoom:(i+1)*zoom, j*zoom:(j+1)*zoom] = (0, 255, 0)
		return out
				
