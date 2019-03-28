#! /usr/bin/env python3
import time

def ConllUReader(treebankfile):
	sentence = {}
	for line in treebankfile:
		if line.startswith("#"):
			continue

		# parse sentence end
		elif line.isspace():
			if sentence != {}:
				yield sentence
			sentence = {}

		else:
			data = line.split("\t")
			if '-' in data[0] or '.' in data[0]:	# skip the word
				continue
			sentence[int(data[0])] = {"form": data[1], "lemma": data[2], "upos": data[3], "xpos": data[4], "feats": data[5], "head": int(data[6]), "deprel": data[7], "deps": data[8], "misc": data[9].strip()}

	if sentence != {}:
		yield sentence

def displayArcs(arcs, sentence):
	for arc in arcs:
		h, d, l = arc
		#print(arc)
		print(sentence[h]["form"], "---" + l + "-->", sentence[d]["form"])
	print()

def displayOriginalArcs(sentence):
	for x in sorted(sentence):
		if sentence[x]["head"] > 0:
			print(sentence[sentence[x]["head"]]["form"], "---" + sentence[x]["deprel"] + "-->", sentence[x]["form"])
	print()

def arc_eager_oracle(stack, buffer, sentence):
	
	if stack == []:
		return 'SHIFT'
	
	else:
		
		# COMPLETE CODE HERE (PART 2)
		if sentence[stack[0]]["head"] == buffer[0]:
			return 'LEFT-ARC-' + sentence[stack[0]]["deprel"]
		
		elif sentence[buffer[0]]["head"] == stack[0]:
			
			return 'RIGHT-ARC-' + sentence[buffer[0]]["deprel"]

		else:
			for k in stack:
				if sentence[buffer[0]]["head"] == k or sentence[k] == sentence[buffer[0]]:
					return 'REDUCE'

			else:
				return 'SHIFT'

# combines algorithms 1 and 3
def parse_arc_eager(sentence, transition_list=None):
	stack = []
	buffer = [x for x in sorted(sentence)]
	count = 1
	arcs = []
	left_arc = ["LEFT-ARC-det", "LEFT-ARC-nsubj", "LEFT-ARC-case", "LEFT-ARC-root", "LEFT-ARC-nmod"]
	right_arc = ["RIGHT-ARC-nmod", "RIGHT-ARC-advmod", "RIGHT-ARC-root"]

	while buffer != []:
		# if transition list is given, pick the first transition (Algorithm 1)
		if transition_list is not None:
			
			t = transition_list.pop(0)

			if t == "SHIFT":
				
				count += 1	
				j = buffer.pop(0)
				stack.insert(0,j)

			elif t in right_arc:

				count += 1	
				s = stack[0]
				b = buffer.pop(0)
				stack.insert(0,b)
				arcs.append((s,b,t))

			elif t in left_arc:
				count += 1
				
				s = stack.pop(0)
				b = buffer[0]
				arcs.append((b,s,t))
				
			elif t == "REDUCE":
				
				count += 1
				s = stack.pop(0)
		
		# if no transition list is given, call the oracle (Algorithm 3)
		else:
			
			print("loop" + str(count))
			print("buffer" + str(buffer))
			print("stack" + str(stack))

			t = arc_eager_oracle(stack, buffer, sentence)

			print("transition = " + str(t))
			
			j = buffer[0]
			print("Top of Buff = " + str(j))
			
			if stack != []:
				i = stack[0]
			else:
				i = "none"

			print("Top of Stack = " + str(i) + "\n")

			if t == "SHIFT":
				stack.insert(0,j)
				buffer.pop(0)
				count+=1
			
			elif t in left_arc:
				count +=1
				arcs.append((j,i,t))
				i = stack.pop(0)
				
			elif t in right_arc:
				count += 1	
				stack.insert(0,j)
				arcs.append((i,j,t))
				buffer.pop(0)
			
			elif t == "REDUCE":
				count += 1
				s = stack.pop(0)

	return arcs


# Test procedure for part 1
def test_parse():
	sentence = {
		1: {"form": "the"},
		2: {"form": "cat"},
		3: {"form": "sits"},
		4: {"form": "on"},
		5: {"form": "the"},
		6: {"form": "mat"},
		7: {"form": "today"}
	}
	
	transition_sequence = ["SHIFT", "LEFT-ARC-det", "SHIFT", "LEFT-ARC-nsubj", "SHIFT", "SHIFT", "SHIFT", "LEFT-ARC-det", "LEFT-ARC-case", "RIGHT-ARC-nmod", "REDUCE", "RIGHT-ARC-advmod", "REDUCE"]
	
	arcs = parse_arc_eager(sentence, transition_list=transition_sequence)
	displayArcs(arcs, sentence)


# Test procedure for part 2
def test_oracle():
	sentence = {
		1: {"form": "the", "head": 2, "deprel": "det"},
		2: {"form": "dog", "head": 3, "deprel": "nsubj"},
		3: {"form": "sat", "head": 0, "deprel": "root"},
		4: {"form": "on", "head": 6, "deprel": "case"},
		5: {"form": "the", "head": 6, "deprel": "det"},
		6: {"form": "couch", "head": 3, "deprel": "nmod"},
		7: {"form": "yesterday", "head": 3, "deprel": "advmod"}
	}
	
	stack = []
	buffer = [1, 2, 3, 4, 5, 6, 7]
	transition = arc_eager_oracle(stack, buffer, sentence)
	print(transition)
	
	stack = [1]
	buffer = [2, 3, 4, 5, 6, 7]
	transition = arc_eager_oracle(stack, buffer, sentence)
	print(transition)
	
	stack = [3]
	buffer = [6, 7]
	transition = arc_eager_oracle(stack, buffer, sentence)
	print(transition)


# Test procedure for part 3
def test_parse_oracle():
	sentence = {
		1: {"form": "the", "head": 2, "deprel": "det"},
		2: {"form": "dog", "head": 3, "deprel": "nsubj"},
		3: {"form": "sat", "head": 0, "deprel": "root"},
		4: {"form": "on", "head": 6, "deprel": "case"},
		5: {"form": "the", "head": 6, "deprel": "det"},
		6: {"form": "couch", "head": 3, "deprel": "nmod"},
		7: {"form": "yesterday", "head": 3, "deprel": "advmod"}
	}
	
	arcs = parse_arc_eager(sentence)
	print("Predicted arcs:")
	displayArcs(arcs, sentence)
	print("Original arcs:")
	displayOriginalArcs(sentence)




if __name__ == "__main__":
	print("-"*38 + "\n\nLab 3\n\nStudent: Robert Rhys Thomas\n Deadline: 09/01/2019\n\n" + "-"*38)

	print("\nPart 1\n")
	test_parse()
	print("-" *38 + "\n")
	print("Part 2\n")
	test_oracle()
	print("\n" + "-" * 38)
	print("\nPart 3\n")
	test_parse_oracle()
	print("-" *38 + "\n\nEnd\n")
	