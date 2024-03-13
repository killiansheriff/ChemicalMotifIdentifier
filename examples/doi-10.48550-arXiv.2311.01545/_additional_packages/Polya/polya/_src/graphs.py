import itertools

import igraph
import matplotlib.pyplot as plt
import numpy as np


class GraphBase:
    def generate_permutations(self, values, coords):
        signs = list(itertools.product([1, -1], repeat=len(values)))
        permutations = []
        for sign in signs:
            value = tuple(val * s for val, s in zip(values, sign))
            permutations.extend(list((itertools.permutations(value, 3))))
        coords = np.vstack((coords, np.array(list(set(permutations)))))

        return coords

    def get_edges(self, vertexpositions, nn_dst, atol=0.1):
        # Subtract each point from all the other points
        diff = vertexpositions[:, np.newaxis, :] - vertexpositions[np.newaxis, :, :]
        # Compute the norm of the differences along the last axis
        distances = np.linalg.norm(diff, axis=-1)
        # Add edges to the graph
        mask = np.isclose(distances, nn_dst, atol=atol)
        edge_index_0, edge_index_1 = np.where(mask)
        edges = [(edge_index_0[i], edge_index_1[i]) for i in range(len(edge_index_0))]
        edges = np.unique(np.sort(edges), axis=0)
        return edges, distances

    def plot(self, g, save_name):
        # Quick viz plotting
        # import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)

        h = g
        # igraph draw
        # ax1.set_title("Plot with igraph plot")
        layout = h.layout_kamada_kawai()
        igraph.plot(h, layout=layout, target=ax)
        ax.set_title(self.graph_name)
        plt.axis("off")
        plt.savefig(save_name)

    def get_graph(self):
        pass


# SPECIFIC GRAPH IMPLEMENTATION


class Fcc1nnGraph(GraphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_graph(self):
        # Define the vertex positions
        # 1NN atoms (sqrt(2)/2 away)
        coords = self.generate_permutations((0, 1 / 2, 1 / 2), np.zeros((1, 3)))
        vertexpositions = coords[1:]

        g = igraph.Graph()
        g.add_vertices(len(vertexpositions))
        edges, distances = self.get_edges(vertexpositions, 1 / np.sqrt(2))

        g.add_edges(edges)
        g.vs["pos"] = vertexpositions

        return g


class Bcc1nnGraph(GraphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_graph(self):
        vertexpositions = np.array(list(itertools.product([-1, 1], repeat=3))) * 0.5

        # Create an empty igraph graph object
        g = igraph.Graph(directed=False)

        g.add_vertices(len(vertexpositions))

        edges_1nn, distances = self.get_edges(vertexpositions, np.sqrt(3) / 2)

        edges_2nn, _ = self.get_edges(vertexpositions, 1)

        if len(edges_1nn) > 0:
            edges = np.concatenate([edges_1nn, edges_2nn])
        else:
            edges = edges_2nn

        g.add_edges(edges)

        return g


class Fcc1nn2nnGraph(GraphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_graph(self):
        # Central atom (Origin)
        coords = np.zeros((1, 3))
        # 1NN atoms (sqrt(2)/2 away)
        values = (0, 1 / 2, 1 / 2)
        coords = self.generate_permutations(values, coords)

        # 2NN atoms (1.0 away)
        values = (0, 0, 1)
        coords = self.generate_permutations(values, coords)

        # remove central atom
        vertexpositions = coords[1:]

        g = igraph.Graph()
        g.add_vertices(len(vertexpositions))
        edges, distances = self.get_edges(vertexpositions, np.sqrt(2) / 2)
        g.add_edges(edges)
        g.vs["pos"] = vertexpositions

        return g


class Hcp1nnGraph(GraphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_graph(self):
        # Define positions of nearest neighbors relative to central atom at (0, 0, 0)

        # Define the lattice parameters
        a = 1
        c = np.sqrt(8 / 3)

        vertexpositions = np.array(
            [
                # [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [1 / 2, np.sqrt(3) / 2, 0],
                [1 / 2, -np.sqrt(3) / 2, 0],
                [-1 / 2, np.sqrt(3) / 2, 0],
                [-1 / 2, -np.sqrt(3) / 2, 0],
                [0, np.sqrt(3) / 3, 1 / 2],
                [0, np.sqrt(3) / 3, -1 / 2],
                [1 / 2, -np.sqrt(3) / 6, 1 / 2],
                [1 / 2, -np.sqrt(3) / 6, -1 / 2],
                [-1 / 2, -np.sqrt(3) / 6, 1 / 2],
                [-1 / 2, -np.sqrt(3) / 6, -1 / 2],
            ]
        )
        vertexpositions *= np.array([a, a, c])

        # Create an empty igraph graph object
        g = igraph.Graph(directed=False)

        # Add vertices to the graph
        g.add_vertices(len(vertexpositions))
        edges, distances = self.get_edges(vertexpositions, 1, atol=0.1)
        g.add_edges(edges)

        # Set the vertex positions
        g.vs["pos"] = vertexpositions

        return g


class Fcc1nn2nn3nnGraph(GraphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_graph(self):
        # Central atom (Origin)
        coords = np.zeros((1, 3))
        # 1NN atoms (sqrt(2)/2 away)
        values = (0, 1 / 2, 1 / 2)
        coords = self.generate_permutations(values, coords)

        # 2NN atoms (1.0 away)
        values = (0, 0, 1)
        coords = self.generate_permutations(values, coords)

        # 3NN atoms (sqrt(3/2) away)
        values = (1, 1 / 2, 1 / 2)
        coords = self.generate_permutations(values, coords)

        # remove central atom
        vertexpositions = coords[1:]

        g = igraph.Graph()
        g.add_vertices(len(vertexpositions))
        edges, distances = self.get_edges(vertexpositions, np.sqrt(2) / 2)
        g.add_edges(edges)
        g.vs["pos"] = vertexpositions

        return g


class Bcc1n2nnnGraph(GraphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_graph(self):
        vertex_1nn = np.array(list(itertools.product([-1, 1], repeat=3))) * 0.5
        vertex_2nn = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        )

        vertexpositions = np.concatenate((vertex_1nn, vertex_2nn))

        g = igraph.Graph()
        g.add_vertices(len(vertexpositions))
        edges_1nn, distances = self.get_edges(vertexpositions, 0.8660254)
        edges_2nn, _ = self.get_edges(vertexpositions, 1)
        edges = np.concatenate((edges_1nn, edges_2nn))  #!!! NOT SURE

        g.add_edges(edges)
        g.vs["pos"] = vertexpositions
        return g
