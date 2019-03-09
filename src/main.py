import argparse
import networkx as nx
import mirand
from gensim.models import Word2Vec


def parse_args():
    '''
    Parses the MIRand arguments.
    '''
    parser = argparse.ArgumentParser(description="Run MIRand.")

    parser.add_argument('--input-struc', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path for structure-layer')

    parser.add_argument('--input-attr', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path for attributes-layer')

    parser.add_argument('--dataset', nargs='?', default='karate',
                        help='Input graph name for saving files')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p-struc', type=float, default=1,
                        help='Return hyperparameter for Structure-Layer. Default is 1.')

    parser.add_argument('--q-struc', type=float, default=1,
                        help='Inout hyperparameter for Structure-Layer. Default is 1.')

    parser.add_argument('--p-attr', type=float, default=1,
                        help='Return hyperparameter for Attribute-Layer. Default is 1.')

    parser.add_argument('--q-attr', type=float, default=1,
                        help='Inout hyperparameter for Attribute-Layer. Default is 1.')

    parser.add_argument('--weighted-struc', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted Structure-Layer. Default is unweighted.')
    parser.add_argument('--unweighted-struc', dest='unweighted', action='store_false')
    parser.set_defaults(weighted_struc=False)

    parser.add_argument('--weighted-attr', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted Attribute-Layer. Default is weighted.')
    parser.add_argument('--unweighted-attr', dest='unweighted', action='store_false')
    parser.set_defaults(weighted_attr=True)

    parser.add_argument('--directed-struc', dest='directed', action='store_true',
                        help='Structure-Layer Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected-struc', dest='undirected', action='store_false')
    parser.set_defaults(directed_struc=False)

    parser.add_argument('--directed-attr', dest='directed', action='store_true',
                        help='Attribute-Layer Graph is (un)directed. Default is directed.')
    parser.add_argument('--undirected-attr', dest='undirected', action='store_false')
    parser.set_defaults(directed_attr=True)

    return parser.parse_args()


def read_graph():
    '''
    Reads the structure and attribute network in networkx.
    '''
    if args.weighted_attr:
        A = nx.read_edgelist(args.input_attr, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        A = nx.read_edgelist(args.input_attr, nodetype=int, create_using=nx.DiGraph())
        for edge in A.edges():
            A[edge[0]][edge[1]]['weight'] = 1

    if args.weighted_struc:
        S = nx.read_edgelist(args.input_struc, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        S = nx.read_edgelist(args.input_struc, nodetype=int, create_using=nx.DiGraph())
        for edge in S.edges():
            S[edge[0]][edge[1]]['weight'] = 1

    if not args.directed_struc:
        S = S.to_undirected()

    if not args.directed_attr:
        A = A.to_undirected()

    return S, A


def generate_walks(G, start=False):
    #################################
    # Calculate transition probabilities for switching the levels
    print('\nStructure layer trans-prob evaluation started')
    G.get_level_transition_weight(0)
    print('\nAttribute layer trans-prob evaluation started')
    G.get_level_transition_weight(1)

    # print('\nalias 1 started')
    G.preprocess_transition_probs(0)
    # print('\nalias 2 started')
    G.preprocess_transition_probs(1)

    print('\nWalk simulation started')
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    return walks


def learn_embeddings(G):
    walks = [map(str, walk) for walk in generate_walks(G, start=True)]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.save_word2vec_format(args.output)
    return


def main(args):
    print('Running MIRand for %s dataset\n' % args.dataset)
    nx_S, nx_A = read_graph()
    G = mirand.Graph(nx_S, nx_A, args.directed_struc, args.directed_attr, args.p_struc, args.q_struc, args.p_attr, args.q_attr)

    assert len(nx_S.nodes()) == len(nx_A.nodes())
    print("Number of nodes in the graph : %s" % len(nx_S.nodes()))
    print("Number of edges in the structure graph : %s" % len(nx_S.edges()))
    print("Number of edges in the attribute graph : %s" % len(nx_A.edges()))

    learn_embeddings(G)


if __name__ == "__main__":
    args = parse_args()
    main(args)
