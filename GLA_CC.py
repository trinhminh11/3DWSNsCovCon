from random import *
from math import *
import numpy as np
import timeit

from CONSTANT import *
from ImportData import *

import numpy
from numpy import sqrt, dot, cross
from numpy.linalg import norm
from ortools.linear_solver import pywraplp

seed(random_seed)

class Vertex(object):
	def __init__(self, T, p):
		self.T = T
		self.v = T.v
		self.neigh = []
		self.q = T.q
		self.p = p

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

def solve(n, regions, q):
    """
    n (int): number of targets
    regions (list): list of region to place sensor. Example: [[1,2,3],[3,4]]
    q (list): priority vector
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    x=[[]]*len(regions)
    for i in range(len(regions)):
        x[i] = solver.IntVar(0,solver.infinity(),' ')
    for j in range(n):
        solver.Add(solver.Sum([x[i] for i in range(len(regions)) if j in regions[i]]) >= q[j])
    M = solver.Sum(x)
    opjective = solver.Minimize(M)
    solver.Solve()
    #for i in range(len(regions)):
    #    print(int(x[i].solution_value()),end=' ')
    return [int(x[i].solution_value()) for i in range(len(regions))]

# Find the intersection of three spheres                 
# P1,P2,P3 are the centers, r1,r2,r3 are the radii       
# Implementaton based on Wikipedia Trilateration article.                              
def trilaterate(P1,P2,P3,r1,r2,r3):                      
	v12 = [P2[i]-P1[i] for i in range(3)]
	d = norm(v12)
	e_x = v12/norm(v12)
	v13 = [P3[i]-P1[i] for i in range(3)]                                       
	i = dot(e_x,v13)                                   
	temp3 = v13 - i*e_x                                
	e_y = temp3/norm(temp3)                              
	e_z = cross(e_x,e_y)                                 
	j = dot(e_y,v13)                                   
	x = (r1*r1 - r2*r2 + d*d) / (2*d)                    
	y = (r1*r1 - r3*r3 -2*i*x + i*i + j*j) / (2*j)       
	temp4 = r1*r1 - x*x - y*y                            
	if temp4 < 0:                                          
		return False, False
	z = sqrt(temp4)                                      
	p_12_a = P1 + x*e_x + y*e_y + z*e_z                  
	p_12_b = P1 + x*e_x + y*e_y - z*e_z   
	return list(p_12_a), list(p_12_b)



class Intersection_Point(object):
	def __init__(self, v, parent, is_3D = True):
		self.v = v
		self.parent = parent
		self.cover = []
		self.is_3D = is_3D

	def is_cover(self, D):
		if D not in self.cover:
			if dist(self.v, D.v) <= D.R  or D in self.parent:
				return True

		return False

	def is_remove(self, rD):
		if self.parent[0] in rD or self.parent[1] in rD or self.parent[2] in rD:
			return True

		return False

	def remove_cover(self, rD):
		for r in rD:
			if r in self.cover:
				self.cover.remove(r)

class Sphere(object):
	def __init__(self, T, R, index):
		self.T = T
		self.v = T.v
		self.q = T.q
		self.R = R
		self.index = index
		self.pair = []
		self.intersections = []
		self.best_point = []
		self.alone = False

	def find_best_point(self):
		self.intersections.sort(reverse = True, key = lambda x: len(x.cover))
		self.best_point.append(self.intersections[0]) 

		for i in range(1, len(self.intersections)):
			if self.intersections[i].cover == self.intersections[0].cover:
				#point B
				self.best_point.append(self.intersections[i])
				break



#Finding sensors
def GLA(T, Rs):
	n = len(T)
	D = [Sphere(T[i], Rs, i) for i in range(n)] #set of Sphere
	D.sort(key = lambda x: x.q)
	S = [] #set of sensor
	GS = [[] for i in range(n)] #set of sensor that target Ti cover



	#find triad
	for i in range(n-2):
		for j in range(i+1, n-1):
			for k in range(j+1, n):
				p1, p2 = trilaterate(D[i].v, D[j].v, D[k].v, Rs, Rs, Rs)
				if p1 and p2:
					parent = (D[i], D[j], D[k])
					for child in parent:
						child.intersections.append(Intersection_Point(p1, parent))
						child.intersections.append(Intersection_Point(p2, parent))

	#find pair
	for i in range(n):
		if len(D[i].intersections) == 0:
			for j in range(n):
				if i != j:
					if dist(D[i].v, D[j].v) <= 2*Rs:
						parent = (D[i], D[j])
						x = (D[i].v[0]+D[j].v[0])/2
						y = (D[i].v[1]+D[j].v[1])/2
						z = (D[i].v[2]+D[j].v[2])/2
						D[i].intersections.append(Intersection_Point((x,y,z), parent))
						D[i].intersections.append(Intersection_Point((x,y,z), parent))

	for Di in D:
		if len(Di.intersections) > 0:
			for point in Di.intersections:
				for Dj in D:
					if point.is_cover(Dj):
						point.cover.append(Dj)
				point.cover.sort(key = lambda x: x.index)
		else:
			Di.alone = True
			Di.intersections.append(Intersection_Point(Di.v, Di))
			Di.intersections.append(Intersection_Point(Di.v, Di))
			for point in Di.intersections:
				point.cover.append(Di)

		Di.find_best_point()


	Regions = []
	Regions_cover = []
	Regions_cover_index = []
	for Di in D:
		if Di.best_point[0].cover not in Regions_cover:
			Regions.append(Di.best_point)
			Regions_cover.append(Di.best_point[0].cover)


	for i in range(len(Regions_cover)):
		Regions_cover_index.append([])
		for j in range(len(Regions_cover[i])):
			Regions_cover_index[i].append(Regions_cover[i][j].index)

	Q = [T[i].q for i in range(len(T))]

	x = solve(n, Regions_cover_index, Q)

	S = []

	for i in range(len(Regions)):
		for j in range(x[i]):
			A = Regions[i][0].v
			B = Regions[i][1].v
			middle = [(A[0]+B[0])/2, (A[1]+B[1])/2, (A[2]+B[2])/2]
			tempS = Sensor(middle, Rs, [])
			S.append(tempS)
							
	return S


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
			T = [Target(Targets[j][k], Qs[j][k], []) for k in range(n)]


			starttime = timeit.default_timer()
			C = Cluster(T, Rs)
			S = []
			Tc = []
			for ii in range(len(C)):
				Tc.append([])
				for jj in range(len(C[ii])):
					Tc[ii].append(C[ii][jj])

			for ii in range(len(C)):
				Sq = GLA(Tc[ii], Rs)
				S += Sq

			average_S[change[i]][j] += len(S)
			endtime = timeit.default_timer()
			average_runtimeCov[change[i]][j] += (endtime-starttime)



		average_S[change[i]][j] = round(average_S[change[i]][j]/data_num)
		average_runtimeCov[change[i]][j] = round(average_runtimeCov[change[i]][j]/data_num, 5)

		exportData(average_S, average_runtimeCov, Dataset, file, "GLA_CC", H, i)

if __name__ == "__main__":
	main()
