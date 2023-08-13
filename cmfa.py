from CONSTANT import *
from random import *
from munkres import Munkres
from math import *
from SPARTA_CC import *
import networkx as nx

from collections import defaultdict

class Graph:
	def __init__(self, vertices):
		self.vertices = vertices
		self.graph = [[0] * vertices for _ in range(vertices)]

	def add_edge(self, u, v, capacity = 1):
		self.graph[u][v] = capacity

	def bfs(self, source, sink, parent, graph):
		visited = [False] * self.vertices
		queue = [source]
		visited[source] = True

		while queue:
			u = queue.pop(0)
			for v in range(self.vertices):
				if not visited[v] and graph[u][v] > 0:
					queue.append(v)
					visited[v] = True
					parent[v] = u
					if v == sink:
						return True

		return False

	def ford_fulkerson(self, source, sink):
		self.temp_graph = [[self.graph[i][j] for j in range(self.vertices)] for i in range(self.vertices)]

		parent = [-1] * self.vertices
		max_flow = 0

		while self.bfs(source, sink, parent, self.temp_graph):
			path_flow = float("Inf")
			s = sink

			while s != source:
				path_flow = min(path_flow, self.temp_graph[parent[s]][s])
				s = parent[s]

			max_flow += path_flow

			v = sink
			while v != source:
				u = parent[v]
				self.temp_graph[u][v] -= path_flow
				self.temp_graph[v][u] += path_flow
				v = parent[v]

		return max_flow

class Vertex:
	def __init__(self, V, index):
		self.V = V
		self.deg = 0
		self.index = index

class Edge:
	def __init__(self, A, B):
		self.V1 = A
		self.V2 = B
		self.dist = dist(A.V.Center.v, B.V.Center.v)

class Cluster:
	def __init__(self, Center, Targets, Qmax):
		self.Center = Center
		self.Targets = Targets
		self.e = -1
		if type(Center) == Base:
			self.e = Qmax
		else:
			for Ti in [Center] + self.Targets:
				if Ti.q > self.e:
					self.e = Ti.q

	def insert_anchor_node(self, S, Rn, Rs):
		for i in range(self.e-self.Center.q):
			tempS = Sensor(self.Center.v, Rs, [self.Center])
			self.Center.Sensors.append(tempS)
			S.append(tempS)
			Rn.append(self.Center.v)

	def Put_relay(self, S, Rn, Rc, Rs):

		for Si in self.Center.Sensors:
			c = dist(Si.v, self.Center.v)
			add = int((c-0.0001)//(Rc))
			for j in range(add):
				x = Si.v[0] + (j+1)*(self.Center.v[0]-Si.v[0])/(add+1)
				y = Si.v[1] + (j+1)*(self.Center.v[1]-Si.v[1])/(add+1)

				sensor = (x, y)
				Rn.append(sensor)

		ans = []
		self.insert_anchor_node(S, Rn, Rs)

		for Ti in self.Targets:
			used = []
			for Sij in Ti.Sensors:
				for Sk in self.Center.Sensors:
					if Sk not in Ti.Sensors and Sk not in used:
						ans.append([Sij, Sk])
						used.append(Sk)
						break


		for Si, Sj in ans:
			c = dist(Si.v, Sj.v)
			add = int((c-0.0001)//(Rc))
			for j in range(add):
				x = Si.v[0] + (j+1)*(Sj.v[0]-Si.v[0])/(add+1)
				y = Si.v[1] + (j+1)*(Sj.v[1]-Si.v[1])/(add+1)
				z = Si.v[2] + (j+1)*(Sj.v[2]-Si.v[2])/(add+1)

				sensor = (x, y, z)
				Rn.append(sensor)

		return ans

def Clustering(T, Rcl, Qmax):
	n = len(T)
	C = []
	used = []

	while True:
		maxneigh = float("-inf")

		bestT = None
		bestNeighs = []

		for Ti in T:
			if Ti not in used:
				center = Ti
				neighbours = []
				for Tj in T:
					if Ti != Tj:
						if Tj not in used:
							if dist(Ti.v, Tj.v) <= Rcl:
								neighbours.append(Tj)

					if len(neighbours) > maxneigh:
						maxneigh = len(neighbours)
						bestT = center
						bestNeighs = neighbours

		if bestT == None:
			break

		C.append(Cluster(bestT, bestNeighs, Qmax))
		used.append(bestT)
		used += bestNeighs

	theta = np.linspace( 0 , 2 * np.pi , 150)

	for Ci in C:
		a = Rcl * np.cos( theta ) + Ci.Center.v[0]
		b = Rcl * np.sin( theta ) + Ci.Center.v[1]

	return C

def Construct_E(C, base, Qmax):
	B = Cluster(base, [], Qmax)
	V = [Vertex(B, 0)] + [Vertex(C[i], i+1) for i in range(len(C))]
	# V = [Vertex(C[i]) for i in range(len(C))]
	L = []
	E = []

	# for Vi in V:
	# 	P = Vi.V.Center.v
	# 	plt.scatter(P[0], P[1], c = 'black', s = 100)

	for i in range(1, len(V)):
		for j in range(i):
			L.append(Edge(V[i], V[j]))

	L.sort(key = lambda x: x.dist)

	for Li in L:
		V1, V2 = Li.V1, Li.V2

		if V1.deg < V1.V.e or V2.deg < V2.V.e:
			E.append(Li)
			V1.deg += 1
			V2.deg += 1


	for r in E:
		L.remove(r)


	for Ei in E:
		P1 = Ei.V1.V.Center
		P2 = Ei.V2.V.Center
		plt.plot([P1.v[0], P2.v[0]], [P1.v[1], P2.v[1]], c = 'black')


	# for i in range(1, len(V)):
	# 	G = nx.DiGraph()
	# 	for Ei in E:
	# 		G.add_edge(Ei.V1, Ei.V2, capacity=1)
	# 		G.add_edge(Ei.V2, Ei.V1, capacity=1)

	# 	source = V[i]
	# 	sink = V[0]

	# 	max_flow_value = nx.maximum_flow(G, source, sink)[0]

	# 	if max_flow_value < V[i].V.e:
	# 		while max_flow_value < V[i].V.e:
	# 			E.append(L[0])
	# 			G.add_edge(L[0].V1, L[0].V2, capacity=1)
	# 			G.add_edge(L[0].V2, L[0].V1, capacity=1)
	# 			max_flow_value = nx.maximum_flow(G, source, sink)[0]
	# 			L.remove(L[0])


	for i in range(1, len(V)):
		G = Graph(len(C)+1)

		Gx = nx.DiGraph()
		for Ei in E:
			G.add_edge(Ei.V1.index, Ei.V2.index)
			G.add_edge(Ei.V2.index, Ei.V1.index)

			Gx.add_edge(Ei.V1, Ei.V2, capacity=1)
			Gx.add_edge(Ei.V2, Ei.V1, capacity=1)

		source = V[i].index
		sink = V[0].index

		sourcex = V[i]
		sinkx = V[0]

		max_flow_value = G.ford_fulkerson(source, sink)

		max_flow_value_Gx = nx.maximum_flow(Gx, sourcex, sinkx)[0]

		if max_flow_value < V[i].V.e:
			while max_flow_value < V[i].V.e:
				E.append(L[0])
				G.add_edge(L[0].V1.index, L[0].V2.index)
				G.add_edge(L[0].V2.index, L[0].V1.index)
				max_flow_value = G.ford_fulkerson(source, sink)
				L.remove(L[0])

	return E

def Put_relay_E(A, B, Rn, Rc):
	P1 = A.v
	P2 = B.v
	c = dist(P1, P2)
	add = int((c-0.0001)//(Rc))
	for j in range(add):
		x = A.v[0] + (j+1)*(B.v[0]-A.v[0])/(add+1)
		y = A.v[1] + (j+1)*(B.v[1]-A.v[1])/(add+1)
		z = A.v[2] + (j+1)*(B.v[2]-A.v[2])/(add+1)

		sensor = (x, y, z)
		Rn.append(sensor)

def CMFA(base, T, S , Rc, Rcl, Rs, Qmax):
	Rn = []
	C = Clustering(T, Rcl, Qmax)


	Rn = []

	for Ci in C:
		Ci.Put_relay(S, Rn, Rc, Rs)


	E = Construct_E(C, base, Qmax)


	for Ei in E:
		Put_relay_E(Ei.V1.V.Center, Ei.V2.V.Center, Rn, Rc)

	return Rn

def main():
	seed(1)

	is_plot = True

	base = Base([0, 0, 0])
	base.Rsz = 0
	H = 1000

	n = 40
	Rs = 40
	Rc = Rs*2
	Rcl = Rs*4
	Qmax = 10
	T = []

	for i in range(n):
		x,y,z = random()*(H-5)+5, random()*(H-5)+5, random()*(H-5)+5
		T.append([x,y,z])

	Q = [randint(1, Qmax) for i in range(n)]

	T = [Target(T[i], Q[i], []) for i in range(len(T))]

	S = SPARTA_CC(T, Rs)



	Rn = CMFA(base, T, S , Rc, Rcl, Rs, Qmax)

	print(len(Rn))

if __name__ == "__main__":
	main()