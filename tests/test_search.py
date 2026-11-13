import unittest
import networkx as nx
from careerfit.search import search


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.g = nx.DiGraph([('s', 'a'), ('a', 'g'), ('s', 'b'), ('b', 'c'), ('c', 'g')])
        nx.set_node_attributes(self.g, {'s': 1, 'a': 90, 'b': 2, 'c': 2, 'g': 1}, 'estimated_hours')

    def test_bfs_fewest_edges(self):
        self.assertEqual(search(self.g, 's', 'g')['path'], ['s', 'a', 'g'])

    def test_weighted_optimality(self):
        for algorithm in ('ucs', 'astar'):
            result = search(self.g, 's', 'g', algorithm)
            self.assertEqual(result['path'], ['s', 'b', 'c', 'g'])
            self.assertEqual(result['route_hours'], 5)

    def test_dfs_deterministic(self):
        self.assertEqual(search(self.g, 's', 'g', 'dfs'), search(self.g, 's', 'g', 'dfs'))

    def test_unreachable(self):
        self.assertFalse(search(self.g, 'g', 's')['found'])

    def test_start_goal(self):
        self.assertEqual(search(self.g, 's', 's')['route_hours'], 0)

    def test_invalid(self):
        with self.assertRaises(ValueError): search(self.g, 'unknown', 'g')
        with self.assertRaises(ValueError): search(self.g, 's', 'g', 'bad')

    def test_cycle_terminates(self):
        self.g.add_edge('a', 's')
        self.assertTrue(search(self.g, 's', 'g', 'dfs')['found'])

    def test_nonpositive_cost_rejected(self):
        self.g.nodes['a']['estimated_hours'] = 0
        with self.assertRaises(ValueError): search(self.g, 's', 'g')
