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
base = [0, 0, 0]

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
	for ii in range(len(C)):
		Tc.append([])
		for jj in range(len(C[ii])):
			Tc[ii].append(C[ii][jj])

	for ii in range(len(C)):
		Sq = Q_Coverage(Tc[ii], Rs)
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

def exportData(average_S, average_runtimeCov, Dataset, file, name, H, change):
	n = Dataset[0][0]
	Rs = Dataset[0][1]
	Q = Dataset[0][2]
	changes = ["n", "R", 'Qmax'] 

	with open(f"{file} {name} change {changes[change]} n{n} Rs{Rs} Q{Q} H{H} data.txt", "w") as f:
		if change == 0:
			f.write("changing n\n")
			for i in range(dataset_num):
				string = f'n = {Dataset[i][0]}, s1-1-{i+1}, {average_S["n"][i]}, {average_runtimeCov["n"][i]}\n'
				f.write(string)
		if change == 1:
			f.write("changing R\n")
			for i in range(dataset_num):
				string = f'R = {Dataset[i][1]}, s1-2-{i+1}, {average_S["R"][i]}, {average_runtimeCov["R"][i]}\n'
				f.write(string)
		if change == 2:
			f.write("changing Qmax\n")
			for i in range(dataset_num):
				string = f'Qmax = {Dataset[i][2]}, s1-3-{i+1}, {average_S["Q"][i]}, {average_runtimeCov["Q"][i]}\n'
				f.write(string)


def main(a,b,c,d,e):
	global H
	Dataset, Targets, Qs, file, i = Import_data(H, data = a, n = b, Rs = c, Qmax = d, change = e)

	average_S = {}
	average_runtimeCov = {}

	average_S['n'] = [0]*dataset_num
	average_runtimeCov['n'] = [0]*dataset_num
	average_S['R'] = [0]*dataset_num
	average_runtimeCov['R'] = [0]*dataset_num
	average_S['Q'] = [0]*dataset_num
	average_runtimeCov['Q'] = [0]*dataset_num


	change = ["n", "R", "Q"]

	for j in range(len(Dataset)): #len(Dataset)
		for run in range(data_num): #data_num
			print(j, run)
			n = Dataset[j][0]
			Rs = Dataset[j][1]

			Qmax = max(Qs[j])

			T = [Target(Targets[j][k], Qs[j][k], []) for k in range(n)]

			starttime = timeit.default_timer()
			S = Q_Coverage(T, Rs)

			average_S[change[i]][j] += len(S)
			endtime = timeit.default_timer()

			average_runtimeCov[change[i]][j] += (endtime-starttime)


		average_S[change[i]][j] = round(average_S[change[i]][j]/data_num)
		average_runtimeCov[change[i]][j] = round(average_runtimeCov[change[i]][j]/data_num, 5)

		exportData(average_S, average_runtimeCov, Dataset, file, "SPARTA", H, i)


if __name__ == "__main__":
	seed(random_seed)
	main(a = 4, b = 100, c = 40, d = 10, e = 1)
	seed(random_seed)
	main(a = 4, b = 400, c = 20, d = 10, e = 2)
	seed(random_seed)
	main(a = 4, b = 400, c = 40, d = 2, e = 3)
