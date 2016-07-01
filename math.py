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

bgun_map = {
	"h1":{"e1⊗f1":2,"e2⊗f2":-1,"e12⊗f12":1,"e23⊗f23":-1,"e123⊗f123":1},
	"h2":{"e1⊗f1":-1,"e2⊗f2":2,"e3⊗f3":-1,"e12⊗f12":1,"e23⊗f23":1},
	"h3":{"e2⊗f2":-1,"e3⊗f3":2,"e12⊗f12":-1,"e23⊗f23":1,"e123⊗f123":1},
	"f1":{"e2⊗f12":-1,"e23⊗f123":-1},
	"f2":{"e1⊗f12":1,"e3⊗f23":-1},
	"f3":{"e2⊗f23":1,"e12⊗f123":1},
	"f12":{"e3⊗f123":-1},
	"f23":{"e1⊗f123":1},
	"f123":{},
}

def printstrs(l):
	for (coef,str) in l:
		print coef, str

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
					result[i] = result[i] + [(1, prefix)]
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
def breakgen(str, i):
	if i < len(str) and str[i].isalpha():
		j = i+1
		while j < len(str) and str[j].isdigit():
			j += 1
		return j
	return -1

# Find the end of a parenthesized expression starting at position i in string s
def matchparen(s, i):
	j = i+1
	level = 0
	while level >= 0:
		if s[j] == '(':
			level += 1
		elif s[j] == ')':
			level -= 1
		j += 1
	return j

# Parse an optional coefficient from string s at position i,
# returning (coef,j) where j is the end of the coefficient (==i if none)
def parsecoef(s, i):
	coef = 1
	j = i
	if s[j] == '+' or s[j] == '-':
		j += 1
	if s[j].isdigit():
		while j < len(s) and s[j].isdigit():
			j += 1
		coef = int(s[i:j])
	else:
		coef = int(s[i:j]+"1")
	return (coef,j)

# Parse a generator term from string s at position i,
# returning (coef,gens,j) where j is the end of the term in string s.
def parseterm(s, i):
	# First break out the coefficient
	(coef, j) = parsecoef(s, i)

	# Next break out the generator(s)
	assert s[j].isalpha() # should have at least one
	k = j
	while k < len(s):
		if s[k].isalpha():
			k = breakgen(s, k)
			assert k > 0
		elif s[k].startswith('⊗'):
			k += len('⊗')
		else:
			break	# not a generator or '⊗'
	gens = s[j:k]

	return (coef,gens,k)

# Invoke function f on each f,e,h token within str.
def foreach_feh(str, f):
	for i in range(len(str)):
		j = breakgen(str, i)
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
			j = breakgen(str, i)
			if j < 0:
				break
			k = breakgen(str, j)
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
			print coef, str, "->", rcoef, rstr
			reslist[0] += [(rcoef, rstr)]
	foreach_feh(str, hit_feh)
	return reslist[0]


class Subspace:
	def __init__(self, s=""):
		self.terms = {}
		i = 0
		while i < len(s):
			(coef, gens, i) = parseterm(s, i)

	def __repr__(self):
		rep = ""
		for gens, coef in self.terms.items():
			if rep != "" and coef >= 0:
				rep += "+"
			if coef == 1:
				rep += gens
			elif coef == -1:
				rep += "-" + gens
			else:
				rep += str(coef) + gens
		return rep

	# Return the coefficient for a given generator string, 0 if none
	def __getitem__(self, gens):
		return self.terms.get(gens, 0)

	# Set the coefficient for a given generator string, delete if coef==0
	def __setitem__(self, gens, coef):
		if coef != 0:
			self.terms[gens] = coef
		elif gens in self.terms:
			del self.terms[gens]

	# Return a new subspace resulting from multiplying self by a scalar
	def __mul__(self, scalar):
		result = Subspace()
		if scalar != 0:
			for gens, coef in self.terms.items():
				result.terms[gens] = coef * scalar
		return result

	def __rmul__(self, scalar):
		return self * scalar

	# Return a subspace resulting from adding two subspaces self and other
	def __add__(self, other):
		result = Subspace()
		def accum(terms):
			for gens, coef in terms.items():
				result[gens] = result[gens] + coef
		accum(self.terms)
		accum(other.terms)
		return result

	# Return a subspace resulting from hitting with an action
	def hit_action(self, action):
		result = Subspace()
		for gens, coef in self.terms.items():
			def hitgen(i, j):
				src = gens[i:j]
				coefdst = action.get(src, 0)
				if coefdst == 0:
					return
				(rcoef,dst) = coefdst
				rcoef *= coef
				rgens = gens[:i] + dst + gens[j:]
				(rcoef,rgens) = sort_fs(rcoef,rgens)
				if rgens != "":
					#print coef, gens, "->", rcoef, rgens
					result[rgens] = result[rgens] + rcoef
			foreach_feh(gens, hitgen)
		return result

	# Return subspace resulting from hitting with a co-boundary operator
	def hit_operator(self, oper):

		def primexpr(basis, i):
			if i >= len(oper):
				return (basis, i)
			elif oper[i] == '(':
				j = matchparen(oper, i)
				(image, k) = primexpr(basis, j)
				return (image.hit_operator(oper[i+1:j-1]), k)
			elif oper[i] == 'f':
				j = breakgen(oper, i)
				assert j > i
				(image, k) = primexpr(basis, j)
				action = actions[oper[i:j]]
				return (image.hit_action(action), k)
			else:
				return (basis, i)

		def multexpr(basis, i):
			(coef, j) = parsecoef(oper, i)
			(image, k) = primexpr(basis, j)
			return (coef * image, k)

		# Sum the results of all terms in the operator
		rsum = Subspace()
		i = 0
		while i < len(oper):	# iterate over terms
			(image, j) = multexpr(self, i)
			rsum = rsum + image
			assert j > i
			i = j
		return rsum

	# Reduce a subspace by eliminating monomials starting with 'f' or 'h'
	def reduce(self):
		result = Subspace()
		for gens, coef in self.terms.items():
			if gens[0] == 'e':
				result[gens] = result[gens] + coef
				continue
			print "reducing",gens
			assert gens[0] == 'f' or gens[0] == 'h'
			i = gens.find("⊗")
			assert i > 0
			j = i+len("⊗")
			conv = bgun_map[gens[:i]]
			tail = gens[j:]
			for cgens, ccoef in conv.items():
				cgens = cgens + tail
				print "c>",ccoef,cgens
				(ccoef, cgens) = sort_fs(ccoef, cgens)
				print "s>",ccoef,cgens
				ccoef = ccoef * coef
				print "m>",ccoef,cgens
				result[cgens] = result[cgens] + ccoef
		return result


# SL4 for v34
v34_weights = [[1,0,0],[0,1,0],[0,0,1]]
v34_strings = [[]] * len(v34_weights)
strs(2, v34_weights, v34_strings)
strs(3, v34_weights, v34_strings)
printweightstrs(v34_weights, v34_strings)

arrows = [
	(0, "-f2f2"),
	(0, "2f1f2-f2f1"),
	(0, "-f3"),
	(1, "2f2f1-f1f2"),
	(1, "-f1f1"),
	(1, "f3f3"),
	(1, "-(2f2f3-f3f2)"),
	(2, "f1"),
	(2, "-(2f3f2-f2f3)"),
	(2, "f2f2"),
]

matdict = {}
coldict = {}
for (weight, operator) in arrows:
	print "Hitting weight", weight, "with", operator
	f2_once = []
	for (coef, gens) in v34_strings[weight]:
		basis = Subspace()
		basis[gens] = coef
		image = basis.hit_operator(operator)
		print basis, "->", image
		image = image.reduce()
		print "reduced:", image
		for igens, icoef in image.terms.items():
			row = matdict.get(igens, {})
			row[gens] = row.get(gens, 0) + icoef
			matdict[igens] = row
			coldict[gens] = 1
print "rows (unique monomials):"
for key in matdict:
	print " ", key
print "cols (basis elements):"
for key in coldict:
	print " ", key
matrix = []
for rowkey, rowdict in matdict.items():
	row = []
	for colkey in coldict:
		row = row + [rowdict.get(colkey, 0)]
	matrix = matrix + [row]
print len(matrix), "x", len(coldict)
print matrix

# SL4 for v46
#v46_weights = [[0,2,1],[0,1,2],[1,0,1],[2,1,0],[1,2,0]]
#v46_strings = [[]] * len(v46_weights)
#strs(3, v46_weights, v46_strings)
#strs(4, v46_weights, v46_strings)
#printweightstrs(v46_weights, v46_strings)

#print "Hitting:"
#for i in range(len(v46_weights)):
#	print "Weight", v46_weights[i]
#	for s in v46_strings[i]:
#		hit(s, action_f1)

