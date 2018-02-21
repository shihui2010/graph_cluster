from scipy.sparse import csr_matrix
import os
import numpy as np
import random
import json


class SparseMatrix(object):
    def __init__(self, sparse_dict, idx_to_parent_id):
        self.idx_map = idx_to_parent_id
        self.graph = sparse_dict
        row_ind, col_ind = list(), list()
        data = list()
        self.dim = 0
        for i in sparse_dict:
            if i >= self.dim:
                self.dim = i + 1
            for j in sparse_dict[i]:
                data.append(1)
                row_ind.append(i)
                col_ind.append(j)
                if j >= self.dim:
                    self.dim = j + 1
        self.sparse_matrix = csr_matrix((data, (row_ind, col_ind)),
                                        shape=(self.dim, self.dim))
        print "sparse matrix built"
        self.R_invs = self.R_inverse(self._R())
        sparse_R_invs = csr_matrix(
            (self.R_invs, (range(self.dim), range(self.dim))),
            shape=[self.dim, self.dim])
        Q1 = sparse_R_invs.dot(self.sparse_matrix)
        Q2 = Q1.dot(self.sparse_matrix)
        self.Q = Q2.dot(sparse_R_invs)
        print "Q matrix calculated"
        self.singular_v = self.v_prime()
        print "V vector got"

    def _R(self):
        """R = diagonal matrix whose diagonal entities are row sum of A_hat * A_hat_T
         rho_i = sum_{j} A_hat{i} . A_hat{j}
            = sum{k} A_hat_{i, k} * (sum{j} A_hat_{j, k}

        as symmetry, sum{j} A_hat_{j, k} = sum{j} A_hat_{k, j}
        as all non-zero elements are 1: sum{j} A_hat_{k, j} = len(graph[k])
        similarly, rho_i = sum{k}{len(graph[k]) where A{i, k} = 1
         return a vector as diagonal entities"""
        diag_elements = [0] * self.dim
        for i in self.graph:
            diag_elements[i] = sum(len(self.graph[k]) for k in self.graph[i])
        return diag_elements

    def R_inverse(self, R):
        """As R being diaginal matrix"""
        return [1.0 / (i + 1e-7) for i in R]

    def v_prime(self):
        """
        Power method for Singular Vactor
        """
        # random initialize v
        v = [random.random() - 0.5 for _ in xrange(self.dim)]
        v_last = [1 for _ in xrange(self.dim)]

        steps = 0
        while True:
            steps += 1
            # make v orthogonal to diagonal vector
            # by changing last element
            except_last_sum = sum([v[i] * self.R_invs[i] for i in xrange(self.dim - 1)])
            v[-1] = (0 - except_last_sum) / self.R_invs[-1]

            # normalize v
            v = np.array(v) / np.sqrt(sum(np.square(v)))

            # update v
            v_new = self.Q.dot(v)

            converge = np.dot(v, v_last) / np.sqrt(sum(np.square(v_last)))
            # print converge
            if converge > 1 - 1e-5 or steps > 2 * np.log(self.dim):
                break
            v_last = v
            v = v_new
        v = np.array(v) / np.sqrt(sum(np.square(v)))
        return v

    def AiAj(self, i, j):
        """
        sine A is binary, symmetry
        """
        product = 0
        for adj_i in self.graph[i]:
            for adj_j in self.graph[j]:
                if adj_i == adj_j:
                    product += 1
        return product

    def top_influencers(self, top=10):
        self_ref = 0.001
        value_vector = np.ones(self.dim) / self.dim
        step = 0
        while True:
            new_values = np.zeros(self.dim)
            for i in xrange(self.dim):
                for j in self.graph[i]:
                    # contribute to neighbors
                    new_values[j] += value_vector[i] / len(self.graph[i])
            new_values = new_values * (1 - self_ref) + self_ref * value_vector
            if new_values.dot(value_vector) < 1e-4:
                break
            elif step >= np.log(self.dim):
                break
        rank = sorted(enumerate(value_vector), key=lambda x: -x[1])
        return rank[:top]

    def split(self):
        id_with_value = zip(range(self.dim), self.singular_v)
        sorted_id_value = sorted(id_with_value, key=lambda x: -x[1])

        # init nominators and denominators
        u = 0   # u0 = 0
        l = 0  # ls = c(empty, all) = 0
        us, ls = list(), list()

        # skip <empty, all> cut
        for i in range(1, self.dim):
            # sum(i+1, n) AiAj
            s1 = sum(self.AiAj(sorted_id_value[i][0], sorted_id_value[j][0])
                     for j in range(i, self.dim))
            # sum(1, i -1) AiAj
            s2 = sum(self.AiAj(sorted_id_value[i][0], sorted_id_value[j][0])
                     for j in range(i - 1))
            # AiAi
            s3 = self.AiAj(sorted_id_value[i][0], sorted_id_value[i][0])
            u += s1 - s2
            l += s1 + s2 + s3
            us.append(u)
            ls.append(l)
            if i == self.dim / 4:
                print "a quarter of scanning done"
            if i == self.dim / 2:
                print "half of scanning done"
            if i == self.dim / 4 * 3:
                print "three quarters of scanning done"
        min_conduct, min_idx = 1 << 31, -1
        for i in range(0, self.dim - 2):
            # i=0 for <one, others> cut
            # self.dim - 3 for <others, one> cut
            conduct = float(us[i]) / min(ls[i], ls[-1] - ls[i])
            if conduct < min_conduct:
                min_conduct = conduct
                min_idx = i + 1

        print min_idx, "at idx", min_idx, "from", self.dim

        # split the matrix
        sorted_id = [i[0] for i in sorted_id_value]
        left_tree = self._split_matrix(sorted_id[:min_idx + 1])
        right_tree = self._split_matrix(sorted_id[min_idx + 1:])
        return left_tree, right_tree

    def _split_matrix(self, indices):
        """
        :param indices: set of indices to split from original graph
        :type return: SparseMatrix
        """
        new2old = dict()
        old2new = dict()
        for i, k in enumerate(indices):
            if self.idx_map:
                new2old[i] = self.idx_map[k]
            else:
                new2old[i] = k
            old2new[k] = i
        new_graph = {i: set() for i in range(len(indices))}
        for i in indices:
            for j in self.graph[i]:
                if j not in indices:
                    # neighbor not in sub-graph
                    continue
                new_graph[old2new[i]].add(old2new[j])
        return SparseMatrix(new_graph, new2old)

    def map_to_root(self, idx):
        if self.idx_map:
            return self.idx_map[idx]
        return idx

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        list_graph = {k: list(v) for k, v in self.graph.items()}
        with open(os.path.join(path, "graph.json"), "w") as fp:
            json.dump(list_graph, fp)
        with open(os.path.join(path, "idx_map.json"), "w") as fp:
            json.dump(self.idx_map, fp)


if __name__ == "__main__":
    from unittest import TestCase
    import unittest
    import numpy as np
    import random

    class TestSparse(TestCase):
        def setUp(self, dim=10):
            self.dense = np.reshape(
                [random.choice([0, 1]) for _ in range(dim * dim)],
                [dim, dim])
            # force matrix to be symmetry
            for i in range(dim):
                self.dense[i][i] = 0
                for j in range(i):
                    self.dense[i][j] = self.dense[j][i]

            # calculate R inverse
            transpose = np.transpose(self.dense)
            matmul = np.matmul(self.dense, transpose)
            dense_r = [sum(i) for i in matmul]
            self.dense_r_invs = [1.0 / (i + 1e-7) for i in dense_r]

            # calculate Q: R_invs . A . A . R_invs
            dmatrix_r_invs = [[0 for _ in xrange(dim)] for _ in xrange(dim)]
            for i in xrange(dim):
                dmatrix_r_invs[i][i] = self.dense_r_invs[i]
            Q1 = np.dot(dmatrix_r_invs, self.dense)
            Q2 = np.dot(Q1, self.dense)
            self.dense_Q = np.dot(Q2, dmatrix_r_invs)

            sparse_dict = {i: set() for i in range(dim)}
            for i, row in enumerate(self.dense):
                for j, element in enumerate(row):
                    if element == 1:
                        sparse_dict[i].add(j)
            self.matrix = SparseMatrix(sparse_dict, None)

        def testMain(self):
            # test r_invs
            dense_r_invs = self.dense_r_invs
            sparse_r_invs = self.matrix.R_invs
            self.assertSequenceEqual(dense_r_invs, sparse_r_invs)

            # test Q
            dim = self.matrix.Q.shape[0]
            sq = self.matrix.Q.todense()
            for i in range(dim):
                srow = sq[i].tolist()[0]
                for j in range(dim):
                    self.assertAlmostEqual(self.dense_Q[i][j], srow[j], 5)

            # test singular vector
            lambdas, vecs = np.linalg.eig(self.dense_Q)
            zipped = zip(lambdas, vecs)
            sec_eigen_value = sorted(zipped, key=lambda x: -x[0])[1][0]
            sparse_v = self.matrix.Q.dot(self.matrix.singular_v)
            eigen_values = list()
            for i, j in zip(self.matrix.singular_v, sparse_v):
                eigen_values.append(j/i)
            # eigen = sorted(eigen_values)[len(eigen_values) / 2]
            # self.assertAlmostEqual(eigen, sec_eigen_value, 2)
            print "dense eigenvalues", sec_eigen_value
            print "sparse eigen values", eigen_values

            # test AiAj
            i, j = 2, 3
            Ai = self.dense[i]
            Aj = self.dense[j]
            self.assertEqual(Ai.dot(Aj), self.matrix.AiAj(i, j))

            # test split
            lt, rt = self.matrix.split()
            print lt.graph
            print rt.graph


    unittest.main()