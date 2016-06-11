#!/usr/bin/python
# coding: utf-8

rank = 3
fs = ["f1", "f2", "f3", "f12", "f23", "f123"]
es = ["e1", "e2", "e3", "e12", "e23", "e123"]
hs = ["h1", "h2", "h3"]

action_f1 = {	"e1":(-1,"h1"), "e12":(-1,"e2"), "e123":(-1,"e23"),
		"f2":(1,"f12"), "f23":(1,"f123"),
		"h1":(2,"f1"), "h2":(-1,"f1") }

action_f2 = {	"e2":(-1,"h2"), "e12":(1,"e1"), "e23":(-1,"e3"),
		"f1":(-1,"f12"), "f3":(1,"f23"),
		"h1":(-1,"f2"), "h2":(2,"f2"), "h3":(-1,"f2")}

action_f3 = {	"e3":(-1,"h3"), "e23":(1,"e2"), "e123":(1,"e12"),
		"f2":(-1,"f23"), "f12":(-1,"f123"),
		"h2":(-1,"f3"), "h3":(2,"f3")}

actions = {"f1":action_f1, "f2":action_f2, "f3":action_f3}

actions["f1"].get("e2",(0,""))

def printstrs(l):
	for s in l:
		print s

def weight(s):
	counts = [0]*rank
	def count(subs, delta):
		for c in subs:
			if c.isdigit():
				counts[int(c)-1] += delta
	split = s.find("⊗")
	count(s[:split], -1)	# e's count negatively
	count(s[split+1:], 1)	# f's count positively
	return counts

# weights is a list of weights of interest
def strs(n_fs, weights, result):
	def all_fs(next, n, prefix):
		if n == 0:
			w = weight(prefix)
			for i in range(len(weights)):
				if weights[i] == w:
					result[i] = result[i] + [prefix]
			return
		for i in range(next, len(fs)):
			str = prefix + fs[i]
			all_fs(i+1, n-1, str)
		return
	for e in es:
		all_fs(0, n_fs, e+"⊗") 
	return result

def printweightstrs(weights, strings):
	for i in range(len(weights)):
		print weights[i]
		printstrs(strings[i])

def foreach_feh(str, f):
	for i in range(len(str)):
		if str[i].isalpha():
			j = i+1
			while j < len(str) and str[j].isdigit():
				j += 1
			f(i, j)

def hit(str, action):
	result = [[]]
	def hit_feh(i, j):
		src = str[i:j]
		coefdst = action.get(src, 0)
		if coefdst == 0:
			return
		(coef,dst) = coefdst
		rstr = str[:i] + dst + str[j:]
		print str, "->", rstr, coef
		result[0] += [(coef, rstr)]
	foreach_feh(s, hit_feh)
	return result


# SL4 for v34
v34_weights = [[1,0,0]]
v34_strings = [[]] * len(v34_weights)
strs(2, v34_weights, v34_strings)
#printweightstrs(v34_weights, v34_strings)

# SL4 for v46
v46_weights = [[0,2,1],[0,1,2],[1,0,1],[2,1,0],[1,2,0]]
v46_strings = [[]] * len(v46_weights)
strs(3, v46_weights, v46_strings)
strs(4, v46_weights, v46_strings)
#printweightstrs(v46_weights, v46_strings)

print "Hitting:"
for s in v46_strings[0]:
	print s
	print hit(s, action_f1)

