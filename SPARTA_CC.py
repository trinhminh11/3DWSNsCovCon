from random import *
from math import *
import numpy as np
import timeit

from CONSTANT import *
from Q_Coverage import *
from Q_Connectivity import *
from cmfa import *
from ImportData import *

seed(random_seed)

class Vertex(object):
	def __init__(self, T, index):
		self.T = T
		self.v = T.v
		self.neigh = []
		self.q = T.q
		self.index = index
		self.p = None

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


def Cluster(T, Rs):
	C = []

	n = len(T)
	parent = [i for i in range(n)]


	V = [Vertex(T[i], i) for i in range(n)]
	E = []
	for i in range(n-1):
		for j in range(i+1, n):
			if dist(V[i].v, V[j].v) <= 2*Rs:
				if find_set(i, parent) == find_set(j, parent):
					continue
				union_sets(i, j, parent)

	for i in range(n):
		V[i].p = find_set(i, parent)

	V.sort(key = lambda x: x.p)
	minp = V[0].p
	maxp = V[-1].p
	Vindex = 0
	for p in range(minp, maxp+1):
		C.append([])

		while Vindex < n and V[Vindex].p == p:
			C[p-minp].append(V[Vindex].T)
			Vindex += 1

	temp = C.count([])
	for i in range(temp):
		C.remove([])

	return C

def SPARTA_CC(T, Rs):
	C = Cluster(T, Rs)
	S = []
	Tc = []
	for i in range(len(C)):
		Tc.append([])
		for j in range(len(C[i])):
			Tc[i].append(C[i][j])

	for i in range(len(C)):
		Sq = Q_Coverage(Tc[i], Rs)
		S += Sq

	return S


def FCSA(Base, S, Rc):
	Rn = []
	for i in range(len(S)):
		P1 = S[i].v
		P2 = Base.v
		c = dist(S[i].v, Base.v)
		add = int((c-1)//(Rc))

		for j in range(add):
			x = P1[0] + (j+1)*(P2[0]-P1[0])/(add+1)
			y = P1[1] + (j+1)*(P2[1]-P1[1])/(add+1)
			z = P1[2] + (j+1)*(P2[2]-P1[2])/(add+1)

			sensor = (x, y, z)
			Rn.append(sensor)

	return Rn



def Plotdata(W, H, T, S, Rs):
	plt.xlabel('width')
	plt.ylabel("height")
	plt.xlim(0, W+Rs)
	plt.ylim(0, H+Rs)
	# fig, ax = plt.subplots()

	theta = np.linspace(0 , 2 * np.pi , 150 ) 
	radius = Rs

	for i in range(len(T)):
		a = T[i].v[0] + radius * np.cos( theta )
		b = T[i].v[1] + radius * np.sin( theta )

		plt.annotate(T[i].q, (T[i].v[0], T[i].v[1]))
		plt.plot(a, b)

	for i in range(len(S)):
		plt.scatter(S[i].v[0], S[i].v[1], s = 50)

	plt.show()



def main():
	global H
	Dataset, Targets, Qs, file, i = Import_data(H)

	average_S = {}
	average_runtimeCov = {}

	average_Rn = {}
	average_runtimeCon = {}

	average_S['n'] = [0]*dataset_num
	average_runtimeCov['n'] = [0]*dataset_num
	average_S['R'] = [0]*dataset_num
	average_runtimeCov['R'] = [0]*dataset_num
	average_S['Q'] = [0]*dataset_num
	average_runtimeCov['Q'] = [0]*dataset_num

	average_Rn['n'] = [0]*dataset_num
	average_runtimeCon['n'] = [0]*dataset_num
	average_Rn['R'] = [0]*dataset_num
	average_runtimeCon['R'] = [0]*dataset_num
	average_Rn['Q'] = [0]*dataset_num
	average_runtimeCon['Q'] = [0]*dataset_num


	change = ["n", "R", "Q"]
	if file == "hanoi":
		base = Base([0, 0, 16.5])
	elif file == "sonla":
		base = Base([0, 0, 945.4])

	alln = []
	allT = []
	allS = []
	allRs = []

	for j in range(len(Dataset)): #len(Dataset)
		for run in range(data_num): #data_num
			print(j, run)
			n = Dataset[j][0]
			Rs = Dataset[j][1]
			Rc = Rs
			Rs = 40
			# Rc = 80
			Rcl = 100

			Qmax = max(Qs[j])

			T = [Target(Targets[j][k], Qs[j][k], []) for k in range(n)]

			starttime = timeit.default_timer()
			S = SPARTA_CC(T, Rs)
			# S = Cluster(T, Rs)

			average_S[change[i]][j] += len(S)
			endtime = timeit.default_timer()
			average_runtimeCov[change[i]][j] += (endtime-starttime)

			starttime = timeit.default_timer()
			Rn = []
			# Rn = FCSA(base, S, Rc)
			# Rn = CMFA(base, T, S , Rc, Rcl, Rs, Qmax)
			Rn = Q_Connectivity(base, T, Rc)

			average_Rn[change[i]][j] += len(Rn)

			endtime = timeit.default_timer()
			average_runtimeCon[change[i]][j] += (endtime-starttime)

		average_S[change[i]][j] = round(average_S[change[i]][j]/data_num)
		average_runtimeCov[change[i]][j] = round(average_runtimeCov[change[i]][j]/data_num, 5)

		average_Rn[change[i]][j] = round(average_Rn[change[i]][j]/data_num)
		average_runtimeCon[change[i]][j] = round(average_runtimeCon[change[i]][j]/data_num, 5)

		exportDataCov(average_S, average_runtimeCov, Dataset, file, "SPARTA_CC", H, i)
		exportDataCon(average_Rn, average_runtimeCon, Dataset, file, "SPARTA_CC", H, i)


if __name__ == "__main__":
	main()