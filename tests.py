import numpy as np
import numpy.testing as npt
import unittest
import sys
import time
import graph_cut
import collage


def random_gm_potentials(M, N, K):
    unary = np.random.rand(N, M, K)
    hor_c = np.random.rand(N, M - 1)
    vert_c = np.random.rand(N - 1, M)
    labels = np.random.permutation(K)
    metric = np.abs(labels[:, None] - labels[None, :])
    return unary, hor_c, vert_c, metric 

def random_labels_and_alpha(M, N, K):
    alpha = np.random.choice(K)
    labels = np.random.choice(K, (N, M))
    return labels, alpha

class TestGM(unittest.TestCase):
    def setUp(self):
        self.M, self.N, self.K = 20, 20, 5
        unary, hor_c, vert_c, metric = random_gm_potentials(self.M,
                                                            self.N,
                                                            self.K)
        self.gm = collage.GM(unary, hor_c, vert_c, metric)
        self.labels, self.alpha = random_labels_and_alpha(self.M,
                                                          self.N,
                                                          self.K)

    def test_energy(self):
        energy = self.gm.energy(self.labels)
        new_labels = self.gm.expand_alpha(self.labels, self.alpha)
        new_energy = self.gm.energy(new_labels)
        self.assertTrue(energy >= new_energy)

    def test_expansion(self):
        alpha_card = np.sum(self.labels == self.alpha)
        new_labels = self.gm.expand_alpha(self.labels, self.alpha)
        new_alpha_card = np.sum(new_labels == self.alpha)
        self.assertTrue(new_alpha_card >= alpha_card)

    def test_graph(self):
        term, edge = self.gm._build_graph(self.labels, self.alpha)
        self.assertTrue(np.all(term >= 0))
        self.assertTrue(np.all(edge >= 0))

    def test_alpha_expansion(self):
        test_mask = np.ones((100, 100), dtype=int)
        test_mask[:50, :50] = 0
        test_mask[50:, :50] = 1
        test_mask[50:, 50:] = 2
        test_mask[:50, 50:] = 3
        unary = np.ones((100, 100, 4), dtype=bool)
        unary[:50, :50, 0] = False
        unary[50:, :50, 1] = False
        unary[50:, 50:, 2] = False
        unary[:50, 50:, 3] = False
        metric = np.invert(np.eye(4, dtype=bool))
        vertC = np.zeros((99, 100))
        horC = np.zeros((100, 99))
        gm = collage.GM(unary, horC, vertC, metric)
        labels = np.random.choice(4, size=(100, 100))
        mask, _, _ = collage.alpha_expansion(gm, labels, display=False)
        self.assertTrue(np.all(mask == test_mask))

if __name__ == '__main__':
    unittest.main()
