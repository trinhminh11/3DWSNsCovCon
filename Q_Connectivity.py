import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from CONSTANT import *
from random import *
import matplotlib.pyplot as plt
from math import *

def find_set(v, parent):

	if v == parent[v]: 
		return v
	p = find_set(parent[v], parent)
	parent[v] = p
	return p

def union_sets(a, b, parent):
	a = find_set(a, parent)
	b = find_set(b, parent)
	if (a != b):
		parent[b] = a


# kruskal algorithm
def Kruskal(S):
	ans = []
	E = []
	parent = [i for i in range(len(S))]

	for i in range(len(S)-1):
		for j in range(i+1, len(S)):
			E.append([dist(S[i].v, S[j].v),i,j])

	E.sort()
	count = 0

	for i in range(len(E)):
		u = E[i][1]
		v = E[i][2]
		if find_set(u, parent) == find_set(v, parent):
			continue
		union_sets(u, v, parent)
		ans.append([u,v])
		count += 1
		if count == len(S)-1:
			break

	return ans
# O(mlog(m))



class Graph():
	def __init__(self, vertices):
		self.V = vertices
		self.graph = [[0 for column in range(vertices)]
					  for row in range(vertices)]
 
	def get_path(self, parent):
		ans = [[parent[i], i] for i in range(1, self.V)]
		return ans
	def minKey(self, key, mstSet):
 
		_min = sys.maxsize
 
		for v in range(self.V):
			if key[v] < _min and mstSet[v] == False:
				_min = key[v]
				min_index = v
 
		return min_index
 
	def primMST(self):
 
		key = [sys.maxsize] * self.V
		parent = [None] * self.V  
		key[0] = 0
		mstSet = [False] * self.V
 
		parent[0] = -1 
 
		for cout in range(self.V):
 
			u = self.minKey(key, mstSet)
 
			mstSet[u] = True
 
			for v in range(self.V):
 
				if self.graph[u][v] > 0 and mstSet[v] == False \
				and key[v] > self.graph[u][v]:
					key[v] = self.graph[u][v]
					parent[v] = u
 
		return self.get_path(parent)

#Q_Connectivity Constraint
def Q_Connectivity(base, T, Rc):
	#base: Coordinates of base
	#GS = {GS1, GS2, ..., GSm} which GSi = set of sensor that covered Target i
	#Rc: Radius of Relay Nodes

	# arange GS
	T.sort(reverse = True, key = lambda x: len(x.Sensors))
	Qmax = T[0].q

	for i in range(1, len(T)):
		j = 0
		while j < len(T[i].Sensors):
			if T[i].Sensors[j] != 0:
				for k in range(i):
					if T[i].Sensors[j] in T[k].Sensors:
						l = T[k].Sensors.index(T[i].Sensors[j])
						T[i].Sensors[j] = 0
						if l < len(T[i].Sensors):
							T[i].Sensors[j], T[i].Sensors[l] = T[i].Sensors[l], T[i].Sensors[j]
							j-=1
						break
			j += 1


	# O(n^2*qmax)

	#devide paths
	paths = []
	for q in range(T[0].q):
		paths.append([])

		for i in range(len(T)):
			if q >= len(T[i].Sensors):
				break
			if T[i].Sensors[q] != 0:
				paths[q].append(T[i].Sensors[q])
	#O(n*qmax)


	Vs = []


	#do Kruskal for each path
	for q in range(T[0].q):

		# Vs.append([[base]+paths[q], Kruskal([base]+paths[q])])

		temp_S = [base] + paths[q]
		temp_n = len(temp_S)
		g = Graph(temp_n)
		g.graph = [[dist(temp_S[i].v, temp_S[j].v) for j in range(temp_n)] for i in range(temp_n)]

		Vs.append([[base]+paths[q], g.primMST()])
		
	#O(nlog(n)*qmax)

	#compute number of relay nodes
	Rn = []
	for q in range(len(Vs)):
		P = Vs[q][0]
		E = Vs[q][1]

		for i in range(len(Vs[q][0])-1):
			P1 = P[E[i][0]].v
			P2 = P[E[i][1]].v
			c = dist(P1, P2)
			add = int((c-1)//(Rc))

			for j in range(add):
				x = P1[0] + (j+1)*(P2[0]-P1[0])/(add+1)
				y = P1[1] + (j+1)*(P2[1]-P1[1])/(add+1)
				z = P1[2] + (j+1)*(P2[2]-P1[2])/(add+1)

				sensor = (x, y, z)
				Rn.append(sensor)
			
	#O(qmax*n)

	return Rn

#O(n^2*qmax)

def Import(file):
	with open(f"{file}.txt", "r") as f:
		for _1 in range(dataset_num):
			for _2 in range(data_num):
				n = int(f.readline())
				Rs = int(f.readline())
				f.readline()
				T = []
				for i in range(n):
					x,y,z,q = list(map(float, f.readline().split(",")))
					q = int(q)
					T.append([x,y,z,q])
				S = []
				m = int(f.readline()[3:])
				for i in range(m):
					x,y,z = list(map(float, f.readline().split(",")))
					S.append(x,y,z)

def main():
	global n, Rs, Rc, Rcl, Qmax, is_plot
	seed(1)

	is_plot = False

	base = Base([0, 0, 0])
	base.Rsz = 0
	H = 1000

	n = 400
	Rs = 40
	Rc = Rs*2
	Qmax = 10
	T = []

	for i in range(n):
		x,y,z = random()*(H-5)+5, random()*(H-5)+5, random()*(H-5)+5
		T.append([x,y,z])

	Q = [randint(1, Qmax) for i in range(n)]

	T = [Target(T[i], Q[i], []) for i in range(len(T))]

	Rn = Q_Connectivity(base, T, Rc)

	print(len(Rn))

	
if __name__ == "__main__":
	# main()
	pass

