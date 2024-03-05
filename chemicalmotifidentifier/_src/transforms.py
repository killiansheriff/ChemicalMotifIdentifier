import numpy as np
import torch
from simplex import Simplex
from torch_geometric.transforms import BaseTransform


class RemoveCentralNode(BaseTransform):
    def __init__(self):
        pass

    def __call__(
        self,
        data,
    ):
        data.central_atom = torch.argmax(data.node_attr[0])
        data.node_attr = data.node_attr[1:]
        data.node_input = data.node_input[1:]

        mask = (data.edge_index[0] != 0) & (data.edge_index[1] != 0)
        data.edge_vec = data.edge_vec[mask]
        data.edge_attr = data.edge_attr[mask]

        edge_index_0 = torch.masked_select(data.edge_index[0], mask) - 1
        edge_index_1 = torch.masked_select(data.edge_index[1], mask) - 1
        data.edge_index = torch.stack((edge_index_0, edge_index_1))

        return data


class AddEdges(BaseTransform):
    def __init__(self, nneigh):
        self.nneigh = nneigh

    def __call__(self, data):
        vertexpositions = data.edge_vec[: self.nneigh]
        diff = vertexpositions[:, np.newaxis, :] - vertexpositions[np.newaxis, :, :]
        distances = torch.linalg.norm(diff, axis=-1)

        second_nn_dst = torch.min(distances[distances != 0])
        mask = torch.where(distances == second_nn_dst)

        data.edge_vec = torch.cat((data.edge_vec, diff[mask]))

        edge_index_0 = torch.cat((data.edge_index[0], mask[0] + 1))
        edge_index_1 = torch.cat((data.edge_index[1], mask[1] + 1))
        data.edge_index = torch.stack((edge_index_0, edge_index_1))

        data.edge_attr = torch.cat(
            (data.edge_attr, torch.ones(mask[0].shape[0], data.edge_attr.shape[1]))
        )

        return data


class ModuloConcentration(BaseTransform):
    def __init__(self, nelement, nneigh):
        self.nelement = nelement
        self.nneigh = nneigh

        ndim = self.nelement - 1
        simplex = Simplex(n_dim=ndim, nneigh=self.nneigh)

        bary_coords, _ = simplex.get_mapping()
        self.generators, counts = simplex.get_generators(bary_coords)

    def __call__(self, data):
        concentration = torch.sum(data.node_attr, dim=0)
        # reverse_onehot = torch.argmax(data.node_attr, dim=1)

        indices = torch.argsort(concentration)
        data.node_attr = data.node_attr[:, indices]
        data.conc = concentration

        return data
