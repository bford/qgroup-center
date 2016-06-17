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

# Find the end of an f, e, or h token starting at position i in str.
# Return -1 if no such token starts at position i.
def break_feh(str, i):
	if i < len(str) and str[i].isalpha():
		j = i+1
		while j < len(str) and str[j].isdigit():
			j += 1
		return j
	return -1

# Invoke function f on each f,e,h token within str.
def foreach_feh(str, f):
	for i in range(len(str)):
		j = break_feh(str, i)
		if j > 0:
			f(i, j)

# Create a dictionary ordering the elements of an array
def orddict(l):
	dict = {}
	for i in range(len(l)):
		dict[l[i]] = i
	return dict

# Create order dictionaries for f list
ford = orddict(fs)

# Sort the f-tokens into standard order, adjusting coefficient sign as needed,
# and dropping terms containing duplicate f-tokens.
def sort_fs(coef, str):
	again = True
	while again:
		again = False
		i = str.index('⊗')+len('⊗')
		while i < len(str):
			# Break out the next two f-tokens
			j = break_feh(str, i)
			if j < 0:
				break
			k = break_feh(str, j)
			if k < 0:
				break
			fa = str[i:j]
			fb = str[j:k]
			if fa == fb:
				return (0,"")	# whole term gets dropped
			if ford[fa] > ford[fb]:
				str = str[:i] + fb + fa + str[k:]
				j = i+len(fb)	# fix position of second f
				coef = -coef	# each swap changes coef sign
				again = True	# iterate until settled
			i = j
	return (coef,str)

def hit(coef, str, action, results):
	reslist = [results]
	def hit_feh(i, j):
		src = str[i:j]
		coefdst = action.get(src, 0)
		if coefdst == 0:
			return
		(rcoef,dst) = coefdst
		rcoef *= coef
		rstr = str[:i] + dst + str[j:]
		(rcoef,rstr) = sort_fs(rcoef,rstr)
		if rstr != "":
			print str, "->", rstr, rcoef
			reslist[0] += [(rcoef, rstr)]
	foreach_feh(s, hit_feh)
	return reslist[0]


# SL4 for v34
v34_weights = [[1,0,0],[0,1,0],[0,0,1]]
v34_strings = [[]] * len(v34_weights)
strs(2, v34_weights, v34_strings)
strs(3, v34_weights, v34_strings)
printweightstrs(v34_weights, v34_strings)

print "Hitting:"
f2_once = []
for s in v34_strings[0]:
	hit(1, s, action_f2, f2_once)
print f2_once
print "Hitting twice:"
f2_twice = []
for v in f2_once:
	print v
	hit(coef, s, action_f2, f2_twice)

# SL4 for v46
v46_weights = [[0,2,1],[0,1,2],[1,0,1],[2,1,0],[1,2,0]]
v46_strings = [[]] * len(v46_weights)
strs(3, v46_weights, v46_strings)
strs(4, v46_weights, v46_strings)
#printweightstrs(v46_weights, v46_strings)

#print "Hitting:"
#for i in range(len(v46_weights)):
#	print "Weight", v46_weights[i]
#	for s in v46_strings[i]:
#		hit(s, action_f1)

