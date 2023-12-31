from random import *
from math import *
import numpy as np
import timeit

from CONSTANT import *
from ImportData import *

seed(random_seed)

class Target:
	def __init__(self, v,q):
		self.v = v
		self.q = q
		self.q_ = 0

class Sensor:
	def __init__(self, v):
		self.v = v
		self.t = []

class Vertex(object):
	def __init__(self, t):
		self.t = t
		self.neigh = []
		self.Cneigh = [0]
		self.mc = 0
		self.avail = 0
		self.color = 0

def VC_SD(T, Rs, Q):
	Targets = [Target(T[i], Q[i]) for i in range(len(T))]
	S = []
	while len(Targets) >1:
		Cmax = 0
		n = len(Targets)
		V = [Vertex(Targets[i]) for i in range(n)]
		E = []
		for i in range(n-1):
			for j in range(i+1, n):
				if dist(V[i].t.v, V[j].t.v) >= Rs:
					V[i].neigh.append(V[j])
					V[j].neigh.append(V[i])
					E.append([V[i], V[j]])


		V.sort(reverse = True, key = lambda x: len(x.neigh))



		V[0].color = 1

		for i in range(1, n):
			for j in range(i):
				if V[j] in V[i].neigh:
					c = V[j].color
					V[i].Cneigh.append(c)


			V[i].mc = max(V[i].Cneigh)
			for j in range(1, V[i].mc+1):

				if j not in V[i].Cneigh:
					V[i].avail = j
					break
			if V[i].avail == 0:
				V[i].color = V[i].mc + 1
			else:
				V[i].color = V[i].avail
			if V[i].color > Cmax:
				Cmax = V[i].color

		V.sort(key = lambda x: x.color)

		Vindex = 0
		qmin = 999
		formerI = 0
		latterI = 0
		qminindex = 0

		for c in range(1, Cmax+1):
			formerI = Vindex
			x, y, z = 0, 0, 0
			Ccount = 0
			while Vindex < n and V[Vindex].color == c:
				if V[Vindex].t.q < qmin:
					qmin = V[Vindex].t.q
					qminindex = Vindex
				x += V[Vindex].t.v[0]
				y += V[Vindex].t.v[1]
				z += V[Vindex].t.v[2]
				Ccount += 1
				Vindex += 1
			latterI = Vindex

			for i in range(qmin):
				S.append((x/Ccount,y/Ccount,z/Ccount))

			Tr = []

			for i in range(formerI,latterI):
				V[i].t.q -= qmin
				if V[i].t.q <= 0:
					Tr.append(V[i].t)

			for r in Tr:
				Targets.remove(r)

	return S

def DSD(T, Q, S, Rs):
	n = len(T)
	m = len(S)
	rS = []

	Targets = [Target(T[i], Q[i]) for i in range(n)]
	Sensors = [Sensor(S[i]) for i in range(m)]

	for i in range(n):
		for j in range(m):
			if dist(Sensors[j].v, Targets[i].v) <= Rs:
				Sensors[j].t.append(Targets[i])
				Targets[i].q_ += 1

	for i in range(m):
		erase = True
		for j in range(len(Sensors[i].t)):
			if Sensors[i].t[j].q_ - Sensors[i].t[j].q == 0:
				erase = False
				break
		if erase:
			rS.append(Sensors[i])

	return len(S) - len(rS)



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

	change = ["n", "R", "Q"]

	for j in range(len(Dataset)):
		for run in range(data_num):
			print(j, run)
			n = Dataset[j][0]
			Rs = Dataset[j][1]
			Q = Qs[j]
			T = Targets[j]

			starttime = timeit.default_timer()
			S = VC_SD(T, Rs, Q)
			S = DSD(T, Q, S, Rs)
			average_S[change[i]][j] += S
			endtime = timeit.default_timer()
			average_runtimeCov[change[i]][j] += (endtime-starttime)

		average_S[change[i]][j] = round(average_S[change[i]][j]/data_num)
		average_runtimeCov[change[i]][j] = round(average_runtimeCov[change[i]][j]/data_num, 5)

		exportData(average_S, average_runtimeCov, Dataset, file, "GD_VC_SD", H, i)

if __name__ == "__main__":
	main()
	# pass
