
# partial attempt at using structured generators,
# but decided it's probably more trouble than it's worth

str2gen = {
	"f1": ('f', frozenset({1})),
	"f2": ('f', frozenset({2})),
	"f3": ('f', frozenset({3})),
	"f12": ('f', frozenset({1,2})),
	"f23": ('f', frozenset({2,3})),
	"f123": ('f', frozenset({1,2,3})),
	"e1": ('e', frozenset({1})),
	"e2": ('e', frozenset({2})),
	"e3": ('e', frozenset({3})),
	"e12": ('e', frozenset({1,2})),
	"e23": ('e', frozenset({2,3})),
	"e123": ('e', frozenset({1,2,3})),
	"h1": ('h', frozenset({1})),
	"h2": ('h', frozenset({2})),
	"h3": ('h', frozenset({3})),
}

gen2str = {
	str2gen["f1"]: "f1",
	str2gen["f2"]: "f2",
	str2gen["f3"]: "f3",
	str2gen["f12"]: "f12",
	str2gen["f23"]: "f23",
	str2gen["f123"]: "f123",
}

class Operator(Terms):
	def __init__(self, s=""):
		Terms.__init__(self)
		i = 0
		while i < len(s):
			(coef, gens, i) = self.parsefterm(s, i)
			self[gens] = self[gens] + coef

	def __repr__(self):
		rep = ""
		for gens, coef in self.items():
			if rep != "" and coef >= 0:
				rep += "+"
			if coef == 1:
				rep += ""
			elif coef == -1:
				rep += "-"
			else:
				rep += str(coef)
			for gen in gens:
				rep += gen2str[gen]
		if rep == "":
			rep = "0"
		return rep

	# Return a new subspace resulting from multiplying self by a scalar
	def __mul__(self, scalar):
		result = Operator()
		if scalar != 0:
			for gens, coef in self.items():
				result.terms[gens] = coef * scalar
		return result

	def __rmul__(self, scalar):
		return self * scalar

	# Return a subspace resulting from adding two subspaces self and other
	def __add__(self, other):
		result = Operator()
		def accum(terms):
			for gens, coef in terms.items():
				result[gens] = result[gens] + coef
		accum(self.terms)
		accum(other.terms)
		return result

	def parsefterm(self, s, i):
		(coef, j) = parsecoef(s, i)
		assert s[j].isalpha() # should have at least one generator
		gens = []
		while j < len(s):
			i = j
			j = breakgen(s, j)
			assert j > 0
			gens = gens + [str2gen[s[i:j]]]
		return (coef,tuple(gens),j)

	# Parse a generator into structured representation
	def parsefgen(self, s, i):
		assert(s[i].isalpha())
		j = i+1
		assert(s[j].isdigit())
		while j < len(s) and s[j].isdigit():
			j += 1
		return generators[s[i:j]], j

	# Attempt to rewrite all terms to end in specified suffix monomial
	def rewrite(self, suffix):
		result = Operator()
		for gens, coef in self.items():

			# Find last generator that doesn't match the suffix
			i = len(gens)-1
			j = len(suffix)-1
			while j >= 0 and gens[i] == suffix[j]:
				i -= 1
				j -= 1
			if j < 0:
				# Suffix already matches, no rewrite needed
				result[gens] = result[gens] + coef
				continue

			# Now search backwards for the last suitable singleton
			# that might plausibly be moved into position i
			# (if it doesn't first get digested by a blob)
			k = i-1
			while gens[k] != suffix[j]:
				k -= 1

			# 

			# XXX ...



op = Operator("f1f2f3")
print op

