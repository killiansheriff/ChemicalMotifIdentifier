import os

import matplotlib

matplotlib.use("Agg")  # Activate 'agg' backend for off-screen plotting.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx

from .transforms import AddEdges

plt.style.use(os.path.join(os.path.dirname(__file__), "mplstyle"))


class Plot:
    """Plotting class to plot microstate based on the switch of atom types from the skeleton ones.

    Args:
        structure (str): can be fcc,bcc or hcp

    Example usage:
    ```python
    plot = Plot(structure="fcc")
    fig, ax = plt.subplots()
    plot.plot_ms(new_atom_types=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 4])
    fig.savefig('fcc.pdf')
    ```
    """

    def __init__(
        self,
        structure,
        graph_folder="/home/ksheriff/PAPERS/second_paper/02_1nn_synthetic/data/graphs",
    ):
        """Plotting class to plot microstate based on the switch of atom types from the skeleton ones.

        Args:
            structure (str): can be fcc,bcc or hcp

        Example usage:
        ```python
        plot = Plot(structure="fcc")
        fig, ax = plt.subplots()
        plot.plot_ms(new_atom_types=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 4])
        fig.savefig('fcc.pdf')
        ```
        """
        self.crystal_structure = structure
        self.graph_folder = graph_folder
        # self.colors = np.array(["#FE218B", "#21B0FE", "#FED700", "orange", "purple", "green"])
        self.colors = np.array(
            ["purple", "orange", "#FED700", "#21B0FE", "#FE218B", "green"]
        )

        self.get_skeleton_graph()
        self.convert_to_nx()
        self.get_edge_color()
        self.get_edge_list()
        self.get_edge_style()
        self.get_node_label()

        self.modes_node_pos = {
            "fcc": self.get_node_pos_fcc,
            "bcc": self.get_node_pos_bcc,
            "hcp": self.get_node_pos_hcp,
        }
        self.modes_node_pos[self.crystal_structure]()

        self.node_size = 300
        self.width = 3

        self.node_edge_colors = self.colors

    def set_colors(self, colors):
        self.colors = colors

    def set_node_edge_colors(self, colors):
        self.node_edge_colors = colors

    def set_node_pos(self, node_pos):
        self.node_pos = node_pos

    def set_node_size(self, node_size):
        self.node_size = node_size

    def set_edge_style(self, edge_style):
        self.edge_Style = edge_style

    def set_width(self, width):
        self.width = width

    def set_df(self, df):
        self.df = df

    def get_skeleton_graph(self):
        self.skeleton_graph = torch.load(
            self.graph_folder + f"/{self.crystal_structure}_1nn.pt"
        )

        if self.crystal_structure == "bcc":
            self.skeleton_graph = AddEdges(nneigh=8)(self.skeleton_graph)

    def convert_to_nx(self):
        self.skeleton_graph.colors = self.colors[
            np.argmax(self.skeleton_graph.node_attr, axis=1)
        ]
        self.g = to_networkx(
            self.skeleton_graph,
            to_undirected=True,
            edge_attrs=["edge_vec"],
            node_attrs=["colors"],
        )

    def switch_atom_types(skeleton_graph, new_atom_types):
        # skeleton_graph.node_attr = torch.argmax(skeleton_graph.node_attr ,dim=1)
        skeleton_graph.node_attr = new_atom_types
        return skeleton_graph

    def get_edge_color(self):
        g = self.g
        color_edges = []
        for edges in g.edges():
            if edges[0] == 0 or edges[1] == 0:
                color_edges.append("#231F20")
            else:
                color_edges.append("#231F20")
        self.color_edges = color_edges

    def get_edge_style(self):
        g = self.g
        edge_style = []
        for edges in g.edges():
            if edges[0] == 0:
                edge_style.append("--")
            elif edges[1] == 0:
                edge_style.append(":")
            else:
                edge_style.append("-")

        self.edge_style = edge_style

    def rotate(self, pos, node_idx):
        # angles between 0 and 2
        u = np.array([0, 1]) - pos[0]
        v = pos[node_idx] - pos[0]

        theta = np.arccos(
            np.dot(u, v) / (np.sqrt(np.sum(u**2)) * np.sqrt(np.sum(v**2)))
        )
        rotMatrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        pos = np.array(list(pos.values())) @ rotMatrix.T
        return pos

    def get_node_pos_fcc(self):
        g = self.g
        edge_vec = np.array(list(nx.get_edge_attributes(g, "edge_vec").values()))

        # pos = np.zeros((nneigh + 1, 3))
        # pos[1 : nneigh + 1] = edge_vec[:nneigh]

        pos = nx.kamada_kawai_layout(g)

        pos = self.rotate(pos, node_idx=2)
        self.node_pos = pos

    def get_node_pos_bcc(self):
        pos = nx.spectral_layout(self.g)
        self.node_pos = np.array(list(pos.values()))
        # breakpoint()
        # pos = nx.spring_layout(self.g, k =1,iterations=10000)
        # pos=nx.spring_layout(nx.cubical_graph(), seed=1234)
        # new_pos = {}
        # new_pos[0]=(pos[4]-pos[6])/2
        # for key in pos.keys():
        #     new_pos[key+1]= pos[key]

        # self.node_pos = np.array(list(new_pos.values()))

    def get_node_pos_hcp(self):
        pos = nx.spring_layout(self.g, seed=4444)
        pos = self.rotate(pos, node_idx=3)
        # pos = nx.kamada_kawai_layout(self.g)
        # shells = [[0], [3,12,7,6,11],[1,2,8,5,10,4,9]]
        # pos = nx.shell_layout(self.g, shells)

        self.node_pos = pos

    def get_node_label(self):
        g = self.g
        label = {}
        for i in list(g.nodes.keys()):
            label[i] = i
        self.node_label = label

    def get_edge_list(self):
        g = self.g
        edge_list = g.edges()
        self.edge_list = edge_list

    def switch_atom_types(self, new_atom_types):
        self.colors_nodes = self.colors[new_atom_types]

        # if node type is -1 (no atoms), then will return the last colior, rempving this seffect and putting it to whithe
        self.colors_nodes[np.where(np.array(new_atom_types) == -1)[0]] = "white"

        # list(nx.get_node_attributes(self.g, "colors").values())

    def plot_ms(self, new_atom_types, draw_label=False, set_lim=True):
        self.switch_atom_types(new_atom_types)

        fig, ax = plt.gcf(), plt.gca()
        # plt.subplots(figsize=(5, 5))

        ec = nx.draw_networkx_edges(
            self.g,
            self.node_pos,
            alpha=1,
            edgelist=self.edge_list,
            arrows=False,
            edge_color=self.color_edges,
            style=self.edge_style,
            width=self.width,
        )

        nc = nx.draw_networkx_nodes(
            self.g,
            self.node_pos,
            nodelist=range(len(self.g.nodes())),
            node_color=self.colors_nodes,
            node_size=self.node_size,
            # edgecolors=self.node_edge_colors, To be put back if want edges
        )

        if draw_label:
            nx.draw_networkx_labels(
                self.g, self.node_pos, self.node_label, font_size=14, font_color="green"
            )

        ax.axis("off")
        ax.axis("equal")

        node_pos = self.node_pos
        xmin, xmax = np.min(node_pos[:, 0]), np.max(node_pos[:, 0])
        ymin, ymax = np.min(node_pos[:, 1]), np.max(node_pos[:, 1])
        offset_y = ymax / 8
        offset_x = xmax / 8
        # print((xmin, xmax))
        if set_lim:
            ax.set_xlim(xmin - offset_x, xmax + offset_x)
            ax.set_ylim(ymin - offset_y, ymax + offset_y)

        return fig, ax

    def plot_counts_for_ms(
        self,
    ):
        fig, ax = plt.gcf(), plt.gca()
        ax.plot(range(10), range(10), "o-")

        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
        ax2 = fig.add_axes([left, bottom, width, height])
        self.plot_ms(new_atom_types=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 4])

        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.55, 0.6, 0.2, 0.2]
        ax3 = fig.add_axes([left, bottom, width, height])

        self.plot_ms(new_atom_types=[0, 1, 2, 0, 1, 2, 0, 1, 3, 3, 3, 2, 4])

        return fig, ax
