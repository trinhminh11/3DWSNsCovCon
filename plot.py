from random import *
from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, dot, cross
from numpy.linalg import norm


#Global Variables

H = 2000	  # Region

n_step = 150		  # step of n
Rs_step = 15		  # step of Rs
Qmax_step = 2	  # step of Qmax

random_seed = 1

seed(random_seed)

# Used Classes
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


#Start Coverage Part


# Find the intersection of three spheres                 
# P1,P2,P3 are the centers, r1,r2,r3 are the radii       
# Implementaton based on Wikipedia Trilateration article.                              
def trilaterate(P1,P2,P3,r1,r2,r3): 
	try:                     
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
	except:
		return False, False



class Intersection_Point(object):
	def __init__(self, v, parent):
		self.v = v
		self.parent = parent
		self.cover = []

	def is_cover(self, Sphere):
		if Sphere not in self.cover:
			if dist(self.v, Sphere.v) <= Sphere.R  or Sphere in self.parent:
				return True

		return False

	def is_remove(self, rD):
		if len(self.parent) == 3:
			if self.parent[0] in rD or self.parent[1] in rD or self.parent[2] in rD:
				return True
		if len(self.parent) == 2:
			if self.parent[0] in rD or self.parent[1] in rD:
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

#Finding sensors
def Q_Coverage(T, Rs):
	n = len(T)
	D = [Sphere(T[i], Rs, i) for i in range(n)] #set of Sphere
	D.sort(key = lambda x: x.q)
	S = [] #set of sensor


	intersection_points = []

	#calc intersection points
	#find triad
	triad = []
	for i in range(n-2):
		for j in range(i+1, n-1):
			for k in range(j+1, n):
				p1, p2 = trilaterate(D[i].v, D[j].v, D[k].v, Rs, Rs, Rs)
				if p1 and p2:
					parent = (D[i], D[j], D[k])
					intersection_points.append(Intersection_Point(p1, parent))
					intersection_points.append(Intersection_Point(p2, parent))
					triad += list(parent)

	#find pair
	# for i in range(n):
	# 	if D[i] not in triad:
	# 		for j in range(n):
	# 			if i != j:
	# 				if dist(D[i].v, D[j].v) <= 2*Rs:
	# 					parent = (D[i], D[j])
	# 					x = (D[i].v[0]+D[j].v[0])/2
	# 					y = (D[i].v[1]+D[j].v[1])/2
	# 					z = (D[i].v[2]+D[j].v[2])/2
	# 					intersection_points.append(Intersection_Point((x,y,z), parent))
	# 					intersection_points.append(Intersection_Point((x,y,z), parent))


	for i in range(n-1):
		for j in range(i+1, n):
			if dist(D[i].v, D[j].v) <= 2*Rs:
				parent = (D[i], D[j])
				x = (D[i].v[0]+D[j].v[0])/2
				y = (D[i].v[1]+D[j].v[1])/2
				z = (D[i].v[2]+D[j].v[2])/2
				intersection_points.append(Intersection_Point((x,y,z), parent))
				intersection_points.append(Intersection_Point((x,y,z), parent))
	# O(n^3)

	#calc point cover by intersection_points
	for point in intersection_points:
		for Di in D:
			if point.is_cover(Di):
				point.cover.append(Di)
								
	#O(n^2*(n-1)/2)

	#calc number of sensor
	while len(D)!= 0:
		#add Q sensors to Sphere that don't intersect with any other Sphere
		if len(intersection_points) == 0:
			
			for Di in D:
				for j in range(Di.q):
					xi, yi, zi = Di.v
					t = uniform(0, 2*pi)
					a = random()
					if a > 0.5:
						a -= 0.001
					x, y, z = cos(t)*a*Rs + xi, sin(t)*a*Rs + yi, zi
					sensor = (x,y,z)
					tempS = Sensor(sensor,Rs, [Di.T])
					S.append(tempS)
					Di.T.Sensors.append(S[-1])
					
			D = []

		#calc number of Sphere covered and index of that Sphere
		else:
			#sort set of intersection points in descending order of number of Target covered
			intersection_points.sort(reverse = True, key = lambda x: len(x.cover))

			#point A
			A = intersection_points[0]
			x1, y1, z1 = A.v

			for i in range(1, len(intersection_points)):
				if intersection_points[i].cover == A.cover:
					#point B
					x2, y2, z2 = intersection_points[i].v
					break

			#add Q sensors to S
			A.cover.sort(key = lambda x: x.q)
			minq = A.cover[0].q
			for i in range(minq):
				#place random
				# a = random()
				# x = x1 + a*(x2-x1)
				# y = y1 + a*(y2-y1)

				#place evenly
				
				x = x1 + (i+1)*(x2-x1)/(minq+1)
				y = y1 + (i+1)*(y2-y1)/(minq+1)
				z = z1 + (i+1)*(z2-z1)/(minq+1)

				sensor = (x, y, z)
				tempS = Sensor(sensor,Rs, [])
				S.append(tempS)
				for j in range(len(A.cover)):
					A.cover[j].T.Sensors.append(S[-1])
					S[-1].Targets.append(A.cover[j].T)

			rD = []

			for i in range(len(A.cover)):
				A.cover[i].q -= minq

				if A.cover[i].q <= 0:
					rD.append(A.cover[i])


			#remove sastified Sphere
			i = 0
			while i < len(intersection_points):
				if intersection_points[i].is_remove(rD):
					intersection_points.pop(i)
					i -= 1
				else:
					intersection_points[i].remove_cover(rD)
				i += 1

			for r in rD:
				D.remove(r)
	#O(n^3)
	return S

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

#End Coverage Part

#Start Connectivity Part

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
		Vs.append([[base]+paths[q], Kruskal([base]+paths[q])])
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

	return Rn, Vs

# End Connectivity Part

def import_data(file):
	with open(f"Data\\{file}.asc", "r") as f:
		f.readline()
		f.readline()
		xllcorner = float(f.readline()[9:-1])
		yllcorner = float(f.readline()[9:-1])
		cellsize = float(f.readline()[8:-1])
		NODATA_value = f.readline()
		data_asc = f.readlines()
		data_asc[0] = data_asc[0][13:]
		data_asc[0] = list(map(float, data_asc[0].split()))
		for i in range(1, len(data_asc)):
			data_asc[i] = list(map(float, data_asc[i].split()))
			data_asc[i-1].append(data_asc[i].pop(0))
		data_asc.pop()
		cell = int(H//25)
		data_asc = data_asc[-cell:]
		for i in range(len(data_asc)):
			data_asc[i] = data_asc[i][:cell]
	
	return data_asc

def place_random(n, Qmax,  data_asc):
	Targets = []

	for _ in range(n):
		x, y = random()*H, random()*H
		z = data_asc[int(y//25)][int(x//25)]
		Targets.append([x,y,z])

	Qs = [randint(1,Qmax) for _ in range(n)]

	return Targets, Qs

def main(file, n, Rs, Rc, Qmax):
	if file == "hanoi":
		base = Base([0, 0, 16.5])
	elif file == "sonla":
		base = Base([0, 0, 945.4])
	
	data_asc = import_data(file)
	
	Targets, Qs = place_random(n, Qmax, data_asc)

	Targets = [Target(Targets[i], Qs[i], []) for i in range(n)]

	Sensors = SPARTA_CC(Targets, Rs)

	Relays, Gs = Q_Connectivity(base, Targets, Rc)

	Plot(data_asc, Targets, Sensors, Gs, Relays, base)

def Plot(data_asc, Targets, Sensors, Gs, Relays, base):

	ax = plt.figure().add_subplot(projection='3d')

	def Pscatter(Points, c, marker, name):
		try:
			x, y, z = zip(*[Point.v for Point in Points])
			ax.scatter(x, y, z, c = c, marker = marker, label = name )
		except:
			x, y, z = zip(*[Point for Point in Points])
			ax.scatter(x, y, z, c = c, marker = marker, label = name)

	Pscatter(Targets,  c = 'blue', marker = 'o', name = 'Targets')
	Pscatter(Sensors, c = 'red', marker = 'o', name = 'Sensors')
	Pscatter(Relays, c = 'orange', marker = 'o', name = 'Relays')

	ax.scatter(*base.v, c = 'black', marker = 'o', label = 'Base')

	for Vs, Es in Gs:
		for E in Es:
			P1 = Vs[E[0]].v
			P2 = Vs[E[1]].v
			P = zip(P1, P2)
			plt.plot(*P, c = 'green')
	
	_x = [25*i for i in range(len(data_asc))]
	_y = [25*i for i in range(len(data_asc))]
	_xx, _yy = np.meshgrid(_x, _y)
	x, y = _xx.ravel(), _yy.ravel()

	top = []

	
	bottom = np.zeros_like(top)
	width = depth = 25

	minx = []
	maxx = []

	for i in range(len(data_asc)):
		minx.append(min(data_asc[i]))
		maxx.append(max(data_asc[i]))

	for i in range(len(data_asc)):
		for j in range(len(data_asc[i])):
			top.append(data_asc[i][j]-min(minx) - shift)

	bottom = [min(minx) for i in range(len(top))]
	ax.bar3d(x, y, bottom, width, depth, top, shade=True)
	

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	file = ["bacgiang", "hanoi", "lamdong", "sonla", "thaibinh"]
	shift = 20
	main(file[3], n = 50, Rs = 40, Rc = 80, Qmax = 4)

