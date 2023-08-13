from random import *
from math import *
import numpy as np
import timeit
import collections	

from CONSTANT import *
from Q_Coverage import *
from ImportData import *
from Q_Connectivity import *

seed(random_seed)


class Vertex(object):
	def __init__(self, t):
		self.t = t
		self.neigh = []

class Cluster(object):
	def __init__(self, v1, v2):
		if type(v1) == Vertex:
			if type(v2) == Vertex:
				self.v = [v1, v2]
			if type(v2) == Cluster:
				self.v = [v1] + v2.v
		if type(v1) == Cluster:
			if type(v2) == Vertex:
				self.v = v1.v + [v2]
			if type(v2) == Cluster:
				self.v = v1.v + v2.v
		self.neigh = []



def Method2(T, Rs):
	S = []
	n = len(T)

	V = [Vertex(T[i]) for i in range(n)]
	E = []

	for i in range(n-1):
		for j in range(i+1, n):
			if dist(V[i].t.v, V[j].t.v) <= 2*Rs:
				V[i].neigh.append(V[j])
				V[j].neigh.append(V[i])
				E.append([V[i], V[j]])

	while True:
		count0 = 0
		for Ver in V:
			if len(Ver.neigh) == 0:
				count0 += 1
		if count0 == len(V):
			break

		V.sort(key = lambda x: len(x.neigh))
		index = 0
		while True:
			if len(V[index].neigh) > 0:
				N1 = V[index]
				break
			else:
				index += 1


		N1.neigh.sort(key = lambda x: len(x.neigh))
		N2 = N1.neigh[0]
		mindeg = len(N1.neigh[0].neigh)
		for i in range(1, len(N1.neigh)):
			if len(N1.neigh[i].neigh) == mindeg:
				temp1 = [N1.neigh[i].neigh[j] for j in range(mindeg) if N1.neigh[i].neigh[j] != N1]
				temp2 = [N1.neigh[j] for j in range(len(N1.neigh)) if N1.neigh[j] != N1.neigh[i]]
				if temp1 == temp2:
					N2 = N1.neigh[i]
			else:
				break

		N = Cluster(N1, N2)
		for Ver in V:
			if Ver != N1 and Ver != N2:
				if N1 in Ver.neigh and N2 in Ver.neigh:
					Ver.neigh.remove(N1)
					Ver.neigh.remove(N2)
					Ver.neigh.append(N)
					N.neigh.append(Ver)
				else:
					try:
						Ver.neigh.remove(N1)
					except:
						pass
					try:
						Ver.neigh.remove(N2)
					except:
						pass

		V.remove(N1)
		V.remove(N2)
		V.append(N)

	C = []
	for i in range(len(V)):
		C.append([])
		if type(V[i]) == Cluster:
			for j in range(len(V[i].v)):
				C[i].append(V[i].v[j].t)

		else:
			C[i].append(V[i].t)

	return C


def main():

	global H
	Dataset, Targets, Qs, file, i = Import_data(H)



	average_S = {}
	average_runtimeCov = {}
	average_S['n'] = [0]*dataset_num
	average_runtimeCov['n'] = [0]*dataset_num
	average_S['R'] = [0]*dataset_num
	average_runtimeCov['R'] = [0]*dataset_num
	average_S['Q'] = [0]*dataset_num
	average_runtimeCov['Q'] = [0]*dataset_num

	average_Rn = {}
	average_runtimeCon = {}
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

	change = ["n", "R", "Q"]

	for j in range(len(Dataset)):
		for run in range(data_num):
			print(j, run)
			n = Dataset[j][0]
			Rs = Dataset[j][1]
			Rc = 80
			T = [Target(Targets[j][k], Qs[j][k], []) for k in range(n)]


			starttime = timeit.default_timer()
			C = Method2(T, Rs)
			S = []
			Tc = []
			for ii in range(len(C)):
				Tc.append([])
				for jj in range(len(C[ii])):
					Tc[ii].append(C[ii][jj])

			for ii in range(len(C)):
				Sq = Q_Coverage(Tc[ii], Rs)
				S += Sq

			# S = Method2(T, Rs)

			starttime = timeit.default_timer()
			Rn = []
			# Rn = FCSA(base, S, Rc)
			# Rn = CMFA(base, T, S , Rc, Rcl, Rs, Qmax)
			Rn = Q_Connectivity(base, T, Rc)

			average_Rn[change[i]][j] += len(Rn)

			endtime = timeit.default_timer()
			average_runtimeCon[change[i]][j] += (endtime-starttime)

			average_S[change[i]][j] += len(S)
			endtime = timeit.default_timer()
			average_runtimeCov[change[i]][j] += (endtime-starttime)



		average_S[change[i]][j] = round(average_S[change[i]][j]/data_num)
		average_runtimeCov[change[i]][j] = round(average_runtimeCov[change[i]][j]/data_num, 5)

		average_Rn[change[i]][j] = round(average_Rn[change[i]][j]/data_num)
		average_runtimeCon[change[i]][j] = round(average_runtimeCon[change[i]][j]/data_num, 5)

		exportDataCov(average_S, average_runtimeCov, Dataset, file, "SPARTA_CP", H, i)
		exportDataCon(average_Rn, average_runtimeCon, Dataset, file, "SPARTA_CP", H, i)
	
if __name__ == "__main__":
	# Cluster(1,1)
	main()
