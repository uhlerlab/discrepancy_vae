import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def draw(pdag, colored_set=set(), solved_set=set(), affected_set=set(), nw_ax=None, edge_weights=None, savefile=None, node_label=None):
    """ 
    plot a partially directed graph
    """
    plt.clf()

    p = pdag.nnodes

    if nw_ax is None:
        nw_ax = plt.subplot2grid((4, 4), (0, 0), colspan=12, rowspan=12)

    plt.gcf().set_size_inches(4, 4)

    # directed edges
    d = nx.DiGraph()
    d.add_nodes_from(list(range(p)))
    for (i, j) in pdag.arcs:
        d.add_edge(i, j)

    # undirected edges
    e = nx.Graph()
    try:
        for pair in pdag.edges:
            (i, j) = tuple(pair, length = 10)
            e.add_edge(i, j)
    except:
        print('there are no undirected edges')
    
    # edge color
    if edge_weights is not None:
        color_d = []
        for i,j in d.edges:
            color_d.append(edge_weights[i,j])

        color_e = []
        for i,j in e.edges:
            color_e.append(edge_weights[i, j])
    else:
        color_d = 'k'
        color_e = 'k'


    # plot
    print("plotting...")
    # pos = nx.circular_layout(d)
    pos = graphviz_layout(d, prog='dot')
    nx.draw(e, pos=pos, node_color='w', style = 'dashed',  edge_cmap=plt.cm.Blues, edge_vmin=-0.025, edge_vmax=0.025, edge_color=color_e)
    color = ['w']*p
    for i in affected_set:
        color[i] = 'orange'
    for i in colored_set:
        color[i] = 'y'
    for i in solved_set:
        color[i] = 'grey'
    nx.draw(d, pos=pos, node_color=color, ax=nw_ax, edge_cmap=plt.cm.RdBu_r, edge_vmin=-0.025, edge_vmax=0.025, edge_color=color_d, width=1.5) #  edge_color='blue',
    nx.draw_networkx_labels(d, pos, labels={node: node_label[node] for node in range(p)}, ax=nw_ax, font_size=12.5)

    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    plt.close()


def draw_spectrum(A, B, savefile=None):
    plt.clf()
    plt.figure(figsize=(3,3))
    e_A = np.linalg.eigvalsh(np.matmul(A.T, A))[::-1]
    e_B = np.linalg.eigvalsh(np.matmul(B.T, B))[::-1]
    plt.plot(np.maximum(e_A,0)**0.5, label=r'$(I-B)^{-1}$')
    plt.plot(np.maximum(e_B,0)**0.5, label=r'$B$')
    plt.legend()
    plt.ylabel('eigenvalues')
    plt.xlabel('index')
    plt.title('Spectrum of SCM')
    plt.tight_layout()
    
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    plt.close()