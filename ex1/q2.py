from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import networkx as nx
from pprint import pprint

#brown_ic = wordnet_ic.ic('ic-brown.dat')
corpus_dir = r'./'
corpus_name = r'corpus_ex1'

newcorpus = PlaintextCorpusReader(corpus_dir, corpus_name)  # creates corpus object
corpus_ic = wn.ic(newcorpus, False, 0.0)  # creates ic object for corpus

music_synset = wn.synset('music.n.01')  # root of our tree

G = nx.Graph()

for s in music_synset.hyponyms():
    G.add_edge(music_synset,s)
    for g in s.hyponyms():
        G.add_edge(s, g)

nodes = G.nodes()

with open('dist_matrix.txt', 'w') as f:
    for u in nodes:
        for v in nodes:
            nx.shortest_path_length(G, u, v)
            # f.write(print(u, v, nx.dijkstra_path_length(G, u, v)))
            s = u, v, nx.shortest_path_length(G, u, v)
            f.write(str(s) + "\n")


with open('lin_sim.txt', 'w') as f:
    # pprint(lin_sims, f)
    for u in nodes:
        for v in nodes:
            if not u is v:
                s = u, v, u.lin_similarity(v, corpus_ic)
                f.write(str(s) + "\n")