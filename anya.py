#!/usr/bin/python
# coding: utf-8

#import scipy
#from scipy import linalg, matrix
from sympy.matrices import Matrix

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

commutes = {
	"f1": {"f3", "f12", "f123"},
	"f2": {"f12", "f23", "f123"},
	"f3": {"f1", "f23", "f123"},
	"f12": {"f1", "f2", "f123"},
	"f23": {"f2", "f3", "f123"},
	"f123": {"f1", "f2", "f3", "f12", "f23", "f123"},
}

rewrites = {
	"f1f2": ("f2f1", +1, "f12"),
	"f2f1": ("f1f2", -1, "f12"),
	"f2f3": ("f3f2", +1, "f23"),
	"f3f2": ("f2f3", -1, "f23"),
	"f1f23": ("f23f1", +1, "f123"),
	"f23f1": ("f1f23", -1, "f123"),
	"f12f3": ("f3f12", +1, "f123"),
	"f3f12": ("f12f3", -1, "f123"),
	"f1f2f2f1": "f2f1f1f2",
	"f2f1f1f2": "f1f2f2f1",
	"f2f3f3f2": "f3f2f2f3",
	"f3f2f2f3": "f2f3f3f2",
}

expansions = {
	"f12": ("f1f2", -1, "f2f1"),
	"f23": ("f2f3", -1, "f3f2"),
	"f123": ("f1f23", -1, "f23f1"),
}

def printstrs(l):
	for (coef,str) in l:
		print(coef, str)

# Find the end of an f, e, or h token starting at position i in str.
# Return -1 if no such token starts at position i.
def breakgen(str, i):
	if i < len(str) and str[i].isalpha():
		j = i+1
		while j < len(str) and str[j].isdigit():
			j += 1
		return j
	return -1

# Tokenize a string of generators into a list of individual generators
def tokgens(str):
	result = []
	i = 0
	while i < len(str):
		j = breakgen(str, i)
		result = result + [str[i:j]]
		i = j
	return result

def weight(s):
	counts = [0]*rank
	delta = 0
	for c in s:
		if c == 'f':
			delta = 1	# f's count positively
		elif c == 'e':
			delta = -1	# e's count negatively
		elif c == 'h':
			delta = 0	# h's count neutrally
		elif c.isdigit():
			counts[int(c)-1] += delta
	return tuple(counts)

# weights is a list of weights of interest
def calcbasis(n_fs, weights, result):
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
			print(coef, str, "->", rcoef, rstr)
			reslist[0] += [(rcoef, rstr)]
	foreach_feh(str, hit_feh)
	return reslist[0]


class Terms:
	def __init__(self):
		self.terms = {}

	# Return the coefficient for a given generator string, 0 if none
	def __getitem__(self, gens):
		return self.terms.get(gens, 0)

	# Set the coefficient for a given generator string, delete if coef==0
	def __setitem__(self, gens, coef):
		if coef != 0:
			self.terms[gens] = coef
		elif gens in self.terms:
			del self.terms[gens]

	def items(self):
		return self.terms.items()


class Subspace(Terms):
	def __init__(self, s=""):
		Terms.__init__(self)
		i = 0
		while i < len(s):
			(coef, gens, i) = parseterm(s, i)
			self[gens] = coef

	def __repr__(self):
		rep = ""
		for gens, coef in self.items():
			if rep != "" and coef >= 0:
				rep += "+"
			if coef == 1:
				rep += gens
			elif coef == -1:
				rep += "-" + gens
			else:
				rep += str(coef) + gens
		if rep == "":
			rep = "0"
		return rep

	# Return a new subspace resulting from multiplying self by a scalar
	def __mul__(self, scalar):
		result = Subspace()
		if scalar != 0:
			for gens, coef in self.items():
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

	def iszero(self):
		return len(self.items()) == 0

	# Return a subspace resulting from hitting with an action
	def hit_action(self, action):
		result = Subspace()
		for gens, coef in self.items():
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
		for gens, coef in self.items():
			if gens[0] == 'e':
				result[gens] = result[gens] + coef
				continue
			assert gens[0] == 'f' or gens[0] == 'h'
			i = gens.find("⊗")
			assert i > 0
			j = i+len("⊗")
			conv = bgun_map[gens[:i]]
			tail = gens[j:]
			for cgens, ccoef in conv.items():
				cgens = cgens + tail
				(ccoef, cgens) = sort_fs(ccoef, cgens)
				ccoef = ccoef * coef
				result[cgens] = result[cgens] + ccoef
		return result

	# Find the set of monomial weights a subspace is comprised of
	def weights(self):
		wts = set()
		for gens, coef in self.items():
			wts.add(weight(gens))
		return wts

	# Find the weight of this subspace, assuming it has exactly one
	def weight(self):
		wts = tuple(self.weights())
		assert(len(wts) == 1)
		return wts[0]

	# Expand composite f generators to represent based on singletons
	def expand(self):
		result = Subspace()
		for gens, coef in self.items():
			# XXX
			return

	# Attempt to rewrite all terms to end in specified suffix monomial
	def resuffix(self, suffix):
		#print("resuffix", self, "suffix", suffix)
		suflist = tokgens(suffix)

		# Perform a substitution that produces 2 terms from one
		def subst2(coef, pre, sub2, post, result):
			(r1,s2,r2) = sub2
			g1 = "".join(pre + [r1] + post)
			result[g1] = result[g1] + coef
			g2 = "".join(pre + [r2] + post)
			result[g2] = result[g2] + coef*s2

		def pull(gens, coef, result):
			genlist = tokgens(gens)

			# Find last generator that doesn't match the suffix
			i = len(genlist)-1
			j = len(suflist)-1
			while j >= 0 and genlist[i] == suflist[j]:
				i -= 1
				j -= 1
			if j < 0:
				# Suffix already matches, no rewrite needed
				result[gens] = result[gens] + coef
				return False

			# Now search backwards for the last suitable singleton
			# that might plausibly be moved into position i
			# (if it doesn't first get digested into a blob)
			want = suflist[j]
			k = i-1
			while k >= 0 and genlist[k] != want:
				k -= 1
			if k < 0:
				# Couldn't find such a singleton,
				# so give up and leave this term unchanged.
				result[gens] = result[gens] + coef
				return False

			changed = True

			# Move it right as far as commutativity rules allow
			while k < i and want in commutes[genlist[k+1]]:
				genlist[k] = genlist[k+1]
				k += 1
				genlist[k] = want
			if k == i:
				gens = "".join(genlist)
				result[gens] = result[gens] + coef
				return True

			# We couldn't commute into the desired position,
			# so rewrite to merge it into a (bigger) blob,
			# which will then commute with more stuff
			# so we'll be able to get something else past it later.

			orig = genlist[k] + genlist[k+1]
			(r1,s2,r2) = rewrites[orig]
			#print(" in", "".join(genlist), "item", k)
			#print("  rewrite", orig, "->", r1, s2, r2)
			subst2(coef, genlist[:k], (r1,s2,r2), genlist[k+2:],
				result)
			return True

		def expand(gens, coef, result):
			genlist = tokgens(gens)
			for i in range(len(genlist)):
				if genlist[i] in expansions:
					changed = True
					sub2 = expansions[genlist[i]]
					#print(" in", "".join(genlist),"item",i)
					#print("  expand", genlist[i])
					subst2(coef, genlist[:i], sub2,
						genlist[i+1:], result)
					return True

			# Nothing to expand
			result[gens] = result[gens] + coef
			return False

		orig = self

		# First tactic: pull singletons rightward,
		# building blobs as needed to allow them to commute.
		# Iterate until we can't make any more progress this way.
		changed = True
		while changed:
			result = Subspace()
			changed = False
			for gens, coef in orig.items():
				if pull(gens, coef, result):
					changed = True
			orig = result

		# Now expand any blobs we might have produced
		changed = True
		while changed:
			result = Subspace()
			changed = False
			for gens, coef in orig.items():
				if expand(gens, coef, result):
					changed = True
			orig = result

		return result

	# Compute an operator by massaging it to match a suffix
	# then removing the suffix
	def calcoperator(self, suffix, sign):
		full = self.resuffix(suffix)
		#print("orig:", self, "resuffix", suffix)
		#print("full:", full)
		clip = len(suffix)
		result = Subspace()
		for gens, coef in full.items():
			assert gens.endswith(suffix)
			prefix = gens[:len(gens)-clip]
			result[prefix] = coef * sign
		return result


# SL4 for v34
v34_lengths = [2,3]
v34_weights = [(1,0,0),(0,1,0),(0,0,1)]
v34_arrows = [
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
v34image_weights = [(0,0,0)]
v34image_arrows = [
	(0, "f1"),
	(0, "f2"),
	(0, "f3"),
]

# SL4 for v46
v46_lengths = [3,4]
v46_weights = [(0,2,1),(0,1,2),(1,0,1),(2,1,0),(1,2,0)]
v46_arrows = [		# Anya's hand-computed
	(0, "f3"),
	(0, "-f1f1f1"),
	(0, "3f2f1-2f1f2"),
	(1, "f2"),
	(1, "f1f1"),
	(1, "-(6f3f2f1-4f2f1f3-3f1f3f2+2f1f2f3)"),
	(2, "f1f1f2f2+4f1f2f1f2+2f2f1f2f1-4f2f1f1f2-2f1f2f2f1"),
	(2, "(4f1f3f2-2f3f2f1-2f1f2f3+f2f3f1)"),
	(2, "-f2f2f2"),
#	(2, XXX
	(3, "(6f1f2f3-4f2f1f3-3f1f3f2+2f3f2f1)"),
	(3, "f3f3"),
	(3, "-f2"),
	(4, "(3f2f3-2f3f2)"),
	(4, "f3f3f3"),
	(4, "-f1"),
]
v46_arrows = [		# computed operators (might be equivalent?)
	(0, "f3"),
	(0, "-(f1f1f1)"),
	(0, "3f2f1-2f1f2"),
	(1, "f2"),
	(1, "f1f1"),
	(1, "-(-2f1f3f2+6f3f2f1-4f2f3f1+2f1f2f3-f3f1f2)"),
	(2, "2f2f1f2f1+2f1f2f1f2-2f2f1f1f2+2f1f1f2f2-3f1f2f2f1"), #?
	(2, "f2f3f1+2f3f1f2-2f1f2f3-2f3f2f1+2f1f3f2"),
	(2, "-(f2f2f2)"),
	(2, "-(2f2f3f2f3-2f2f3f3f2+2f3f2f3f2+2f3f3f2f2-3f3f2f2f3)"), #?
	(3, "-3f1f3f2-2f2f3f1+6f1f2f3-2f2f1f3+2f3f2f1"),
	(3, "f3f3"),
	(3, "-f2"),
	(4, "(3f2f3-2f3f2)"),
	(4, "f3f3f3"),
	(4, "-f1"),
]

# SL4 for v58
v58_lengths = [4,5]
#	       s1s3s2  s2s1s3  s3s2s1  s1s2s1  s1s2s3  s3s2s3
v58_weights = [(2,1,2),(1,3,1),(1,2,3),(2,2,0),(3,2,1),(0,2,2)]
v58_arrows = [
	(0, "-(3f1f2-2f2f1)"),
	(0, "f2f2f2"),
	(0, "-(3f3f2-2f2f3)"),
	(1, "(4f2f1f3-2f1f2f3-2f3f2f1+f1f3f2)"),
	(1, "f1f1"),
	(1, "-f3f3"),
	(2, "f2"),
	(2, "-f1"),
	(3, "(6f1f2f3-4f1f3f2-3f2f1f3+2f3f2f1)"),
	(3, "-f3f3f3"),
	(4, "f3"),
	(4, "f2"),
	(5, "f1f1f1"),
	(5, "(6f3f2f1-4f1f3f2-3f2f1f3+2f1f2f3)"),
]

# SL4 for v610
v610_lengths = [5,6]
v610_weights = [(3,2,2),(2,4,2),(1,3,3),(3,3,1),(2,2,3)]
v610_arrows = [
	(0, "f2f2"),
	(0, "(2f3f2-f2f3)"),
	(1, "f1"),
	(1, "f3"),
	(2, "-(f1f1)"),
	(2, "(2f2f1-f1f2)"),
	(3, "-(2f2f3-f3f2)"),
	(3, "-(f3f3)"),
	(4, "-(2f1f2-f2f1)"),
	(4, "f2f2"),
]

graph = [
	{ # d0
		"": [	(+1, "f3"),
			(+1, "f2"),
			(+1, "f1"),
		],
	},
	{ # d1: v34
		"f3": [	(+1, "f2f2f3"),
			(-1, "f3f3f2"),
			(+1, "f1f3"),
		],
		"f2": [	(-1, "f2f2f3"),
			(+1, "f3f3f2"),
			(-1, "f1f1f2"),
			(+1, "f2f2f1"),
		],
		"f1": [	(-1, "f1f3"),
			(+1, "f1f1f2"),
			(-1, "f2f2f1"),
		],
	},
	{ # d2: v46
		"f2f2f3": [
			(+1, "f2f3f3f2"),
			(-1, "f1f1f1f2f2f3"),
			(+1, "f2f2f2f3f1"),
		],
		"f3f3f2": [
			(+1, "f2f3f3f2"),
			(+1, "f3f3f1f1f2"),
			(-1, "f3f3f3f2f2f1"),
		],
		"f1f3": [
			(+1, "f1f1f1f2f2f3"),	#?
			(+1, "f3f3f1f1f2"),
			(-1, "f2f2f2f3f1"),
			(-1, "f3f3f3f2f2f1"),	#?
		],
		"f1f1f2": [
			(+1, "f1f1f1f2f2f3"),
			(+1, "f3f3f1f1f2"),
			(-1, "f1f2f2f1"),
		],
		"f2f2f1": [
			(+1, "f2f2f2f3f1"),
			(+1, "f3f3f3f2f2f1"),
			(-1, "f1f2f2f1"),
		],
	},
	{ # d3
		"f2f3f3f2": [
			(+1, "f1f1f1f2f3f3f2"),
			(+1, "f2f3f3f3f2f2f1"),
		],
		"f1f1f1f2f2f3": [
			(+1, "f1f1f1f2f3f3f2"),
			(+1, "f1f1f2f2f2f3f1"),
		],
		"f3f3f1f1f2": [
			(-1, "f1f1f1f2f3f3f2"),
			(+1, "f2f2f2f3f3f1f1f2"),
			(-1, "f1f3f3f3f2f2f1"),
		],
		"f2f2f2f3f1": [
			(+1, "f2f2f2f3f3f1f1f2"),
			(+1, "f1f1f2f2f2f3f1"),
			(-1, "f2f3f3f3f2f2f1"),
		],
		"f3f3f3f2f2f1": [
			(+1, "f2f3f3f3f2f2f1"),
			(-1, "f1f3f3f3f2f2f1"),
		],
		"f1f2f2f1": [
			(+1, "f1f1f2f2f2f3f1"),
			(-1, "f1f3f3f3f2f2f1"),
		],
	},
	{ # d4
		"f1f1f1f2f3f3f2": [
			(+1, "f2f2f1f1f1f2f3f3f2"),
			(+1, "f1f1f2f3f3f3f2f2f1"),
		],
		"f2f2f2f3f3f1f1f2": [
			(+1, "f2f2f1f1f1f2f3f3f2"),
			(+1, "f2f2f3f3f3f1f2f2f1"),
		],
		"f1f1f2f2f2f3f1": [
			(-1, "f2f2f1f1f1f2f3f3f2"),
			(-1, "f1f1f2f3f3f3f2f2f1"),
		],
		"f2f3f3f3f2f2f1": [
			(-1, "f1f1f2f3f3f3f2f2f1"),
			(+1, "f2f2f3f3f3f1f2f2f1"),
		],
		"f1f3f3f3f2f2f1": [
			(-1, "f1f1f2f3f3f3f2f2f1"),
			(+1, "f2f2f3f3f3f1f2f2f1"),
		],
	},
	{ # d5
		"f2f2f1f1f1f2f3f3f2": [
			(+1, "f1f2f2f3f3f3f1f2f2f1"),
		],
		"f1f1f2f3f3f3f2f2f1": [
			(-1, "f1f2f2f3f3f3f1f2f2f1"),
		],
		"f2f2f3f3f3f1f2f2f1": [
			(-1, "f1f2f2f3f3f3f1f2f2f1"),
		],
	},
]


class Case:
	def __init__(self, slen, name, lengths, weights, arrows):
		self.slen = slen
		self.name = name
		self.lengths = lengths
		self.weights = weights
		self.arrows = arrows

		self.basis = [[]] * len(weights)
		for l in lengths:
			calcbasis(l, weights, self.basis)

		if slen < 3:	# XXX
			self.calcoperators()

	def calcoperators(self):
		level = self.slen
		self.operators = {}
		print("Level",level,"operators:")
		for (src, edges) in graph[level].items():
			w = weight(src)
			print(w)	#, "from", src
			oplist = []
			for (sgn, dst) in edges:
				#print src, sgn, dst
				subs = Subspace(dst).calcoperator(src, 1)
				s = str(subs)
				if sgn < 0:
					s = "-(" + s + ")"
				print("\t", s)	#, "to", dst
				oplist = oplist + [(s, weight(dst))]
			self.operators[w] = oplist
		#print(self.operators)


cases = [
	Case(1, "v34", v34_lengths, v34_weights, v34_arrows),
	Case(2, "v46", v46_lengths, v46_weights, v46_arrows),
	Case(3, "v58", v58_lengths, v58_weights, v58_arrows),
]

def checkcases(level):
	print("Checking cases at level", level)
	def flow(level, subs):
		print("flow", level, subs)
		case = cases[level]
		next = Subspace()
		ncase = cases[level+1]
		for gens, coef in subs.items():
			w = weight(gens)
			for (wt, op) in case.arrows:
				if case.weights[wt] == w:
					base = Subspace()
					base[gens] = coef
					imag = base.hit_operator(op)
					imag = imag.reduce()
					print("  ",base, w, "->", op, "->", imag, imag.weights())
					next = next + imag
		print("next", next)
		print("weights", next.weights())
		return next

	case = cases[level]
	#for w in range(len(case.weights)):
	for w in [1]:
		for (c, g) in case.basis[w]:
			print("Checking weight", case.weights[w], "basis", c, g)
			b = Subspace()
			b[g] = c
			n = flow(level, b)
			f = flow(level+1, n)

#checkcases(0)

def check_parallelograms(slen):
	print("Checking cases at level", slen)
	def flow(slen, depth, base, results):
		#print("flow", slen, base)
		wt = base.weight()
		for (oper, iwt) in cases[slen].operators[wt]:
			imag = base.hit_operator(oper)
			if imag.iszero():
				continue
			assert(imag.weight() == iwt)
			if depth > 1:
				flow(slen+1, depth-1, imag, results)
			else:
				rsub = results.get(iwt, Subspace())
				results[iwt] = rsub + imag

	case = cases[slen]
	for w in range(len(case.weights)):
		for (c, g) in case.basis[w]:
			print("Checking weight", case.weights[w], "basis", c, g)
			b = Subspace()
			b[g] = c
			r = {}
			n = flow(slen, 2, b, r)
			#print("weight",case.weights[w],"result:", r)
			for iwt, sub in r.items():
				assert sub.iszero()

check_parallelograms(0)
#check_parallelograms(1)



def calcmatrix(lengths, weights, arrows):

	def printweightstrs(weights, strings):
		for i in range(len(weights)):
			print(weights[i])
			printstrs(strings[i])

	strings = [[]] * len(weights)
	for l in lengths:
		calcbasis(l, weights, strings)
	printweightstrs(weights, strings)

	matdict = {}
	coldict = {}
	for (weight, operator) in arrows:
		#print("Hitting weight", weight, "with", operator)
		for (coef, gens) in strings[weight]:
			coldict[gens] = 1
			basis = Subspace()
			basis[gens] = coef
			image = basis.hit_operator(operator)
			image = image.reduce()
			#print(basis, "->", image)
			for igens, icoef in image.items():
				row = matdict.get(igens, {})
				row[gens] = row.get(gens, 0) + icoef
				matdict[igens] = row

	'''
	print("rows (unique monomials):")
	for key in matdict:
		print(" ", key)
	print("cols (basis elements):")
	for key in coldict:
		print(" ", key)
	'''

	matrix = []
	for rowkey, rowdict in matdict.items():
		row = []
		for colkey in coldict:
			row = row + [rowdict.get(colkey, 0)]
		matrix = matrix + [row]
	print(len(matrix), "x", len(coldict))
	M = Matrix(matrix)
	r = M.rank()
	#print("nullspace", M.nullspace())
	n = len(coldict)-r
	print("rank", r, "nullspace", n)
	return (M,r,n)

def calcnullspace(name, lengths, kern_weights, kern_arrows,
			img_weights, img_arrows):

	kM,kR,kN = calcmatrix(lengths, kern_weights, kern_arrows)
	iM,iR,iN = calcmatrix(lengths, img_weights, img_arrows)

	print(name, "kernel:", kM.shape, "rank", kR, "nullspace", kN)
	print(name, "image:", iM.shape, "rank", iR, "nullspace", iN)

	print("Answer:", kN-iR)


#calcnullspace("v34", v34_lengths, v34_weights, v34_arrows, v34image_weights, v34image_arrows)

#calcnullspace("v46", v46_lengths, v46_weights, v46_arrows, v34_weights, v34_arrows)

#calcnullspace("v58", v58_lengths, v58_weights, v58_arrows, v46_weights, v46_arrows)

calcnullspace("v610", v610_lengths, v610_weights, v610_arrows, v58_weights, v58_arrows)

