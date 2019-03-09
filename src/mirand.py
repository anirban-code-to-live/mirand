import numpy as np
import networkx as nx
import random


class Graph():
    def __init__(self, nx_S, nx_A, is_directed_struc, is_directed_attr, p_struc, q_struc, p_attr, q_attr):
        self.G = [nx_S, nx_A]
        self.is_directed = [is_directed_struc, is_directed_attr]
        self.p = [p_struc, p_attr]
        self.q = [q_struc, q_attr]
        self.alias_nodes = [None, None]
        self.alias_edges = [None, None]
        self.trans_weight = [None, None]
        self.ct = [0, 0]

    # Modified node2vec walk for multi-layered graph with structure and content
    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            # Decide the layer structure/attribute for the current node
            # taking upper as structure and lower level as attribute
            up = self.trans_weight[1][cur]
            down = self.trans_weight[0][cur]
            pu = up / (up + down)  # probability to select in structure layer
            pd = 1 - pu  # probability to select in attribute layer

            x = random.random()  # random num between 0---1
            if x < pu:  # if pu is large then more chances of Reference being selected
                ind = 0
            else:
                ind = 1
            self.ct[ind] += 1  # to get count which layer the Random walk is in.

            cur_nbrs = sorted(self.G[ind].neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[ind][cur][0], alias_nodes[ind][cur][1])])
                else:
                    prev = walk[-2]
                    if (prev, cur) not in alias_edges[ind]:  # when the edge is not in other graph
                        walk.append(cur_nbrs[alias_draw(alias_nodes[ind][cur][0], alias_nodes[ind][cur][1])])
                    else:
                        e1 = alias_edges[ind][(prev, cur)][0]
                        e2 = alias_edges[ind][(prev, cur)][1]
                        tmp = alias_draw(e1, e2)
                        next = cur_nbrs[tmp]
                        walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G[0]  # we can take any graph as we just need to find the nodes
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            print('Walk iteration :: %s / %s' % (str(walk_iter + 1), str(num_walks)))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
            print('Walk count in Structure layer :: %s  , Attribute layer :: %s' % (str(self.ct[0]), str(self.ct[1])))
            self.ct = [0, 0]
        return walks

    def get_level_transition_weight(self, ind):
        G = self.G[ind]
        mat = nx.to_scipy_sparse_matrix(G)
        if ind == 0:
            avg = 1.0
        else:
            avg = 1.0 * np.sum(mat) / G.number_of_edges()
        if ind == 0:
            print('Threshold for structure layer :: %s' % str(avg))
        else:
            print('Threshold for attribute layer :: %s' % str(avg))
        mat = mat >= avg
        tau = np.sum(mat, axis=1)
        self.trans_weight[ind] = np.log(np.e + tau)

    def get_alias_edge(self, src, dst, ind):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G[ind]
        p = self.p[ind]
        q = self.q[ind]

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self, ind):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G[ind]
        is_directed = self.is_directed[ind]

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1], ind)
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1], ind)
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0], ind)

        self.alias_nodes[ind] = alias_nodes
        self.alias_edges[ind] = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    tmp = q[kk]
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
