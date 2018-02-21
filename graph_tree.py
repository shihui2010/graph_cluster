from sparse_matrix import SparseMatrix
import json
import os
import re
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
prenoun = {"a", "an", "many", "all", "each", "any", "and", "similar",
           "several", "some", "most", "another"}
quantifier_pattern = re.compile("a .+ of ")


def stem(token):
    token = re.sub(quantifier_pattern, "", token)
    stems = list()
    for i in token.split():
        if i in prenoun:
            continue
        if len(i) == 1:
            return ""
        try:
            stems.append(porter_stemmer.stem(i))
        except:
            continue
    try:
        return " ".join(sorted(stems))
    except:
        return None


with open("entity_set.json") as fp:
    entity_set = json.load(fp)

idx = 0
idx2entity = dict()
entity2idx = dict()
graph = dict()
for entity in entity_set:
    if len(entity) > 1:
        idx2entity[idx] = entity
        entity2idx[entity] = idx
        graph[idx] = set()
        idx += 1
print "entities set up", len(graph)
# print entity2idx

with open("KeyRelation.txt") as fp:
    counter = 0
    for line in fp:
        line_split = line.strip().split("\t")
        stemmed_sub = stem(line_split[2])
        if stemmed_sub not in entity2idx:
            continue
        sub_idx = entity2idx[stemmed_sub]
        stemmed_obj = stem(line_split[4])
        if stemmed_obj not in entity2idx:
            continue
        obj_idx = entity2idx[stemmed_obj]
        graph[sub_idx].add(obj_idx)
        graph[obj_idx].add(sub_idx)
        counter += 1
        if counter % 10000 == 0:
            print counter, "edges added"
print "graph pre-loaded"


root = SparseMatrix(graph, None)
print "graph constructed"

leaves = [root]
tree = [leaves]

for level in range(4):
    # 4-levels tree
    lower_leaves = list()
    for idx, leaf in enumerate(leaves):
        print "splitting level %d idx %d" % (level, idx)
        leaf.save("tree/" + str(level) + "_" + str(idx))
        lower_leaves.extend(leaf.split())
    leaves = lower_leaves
    tree.append(leaves)
