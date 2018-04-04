from sparse_matrix import SparseMatrix
import json
import os
import re
from multiprocessing import Pool
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


def influencer(sp, path):
    key_entities = sp.top_influencers(top=100)
    names = dict()
    for index, score in key_entities:
        name = idx2entity[sp.map_to_root(index)]
        names[name] = score
    with open(os.path.join(path, "keys.json"), "w") as fp:
        json.dump(names, fp, indent=4)
    print "saved key influencers at", path


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
root.save("tree/root")
influencer(root, "tree/root")
print "graph constructed"


def split_tree(path_name):
    with open(os.path.join(path_name, "graph.json")) as fp:
        graph = json.load(fp)
    with open(os.path.join(path_name, "idx_map.json")) as fp:
        idx = json.load(fp)
    sparse_matrix = SparseMatrix(graph, idx)
    left_child, right_child = sparse_matrix.split()

    left_child.save(path_name + ".L")
    influencer(left_child, path_name + ".L")
    right_child.save(path_name + ".R")
    influencer(right_child, path_name + ".R")

    children_paths = []
    if left_child.dim > 200:
        children_paths.append(path_name + ".L")
    if right_child.dim > 200:
        children_paths.append(path_name + ".R")
    return children_paths

pool = Pool()
cur_level = ["tree/root"]
for _ in range(9):
    sub_trees = pool.map(split_tree, cur_level)
    leaves = []
    for i in sub_trees:
        leaves.extend(i)
    cur_level = leaves