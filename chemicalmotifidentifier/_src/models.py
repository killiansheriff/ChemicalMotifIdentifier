from typing import Dict, Union

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.v2103.gate_points_message_passing import MessagePassing
from torch_geometric.data import Data
from torch_scatter import scatter


class NetworkForAGraphWithAttributesPeriodic(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        max_radius,
        num_neighbors,
        num_nodes,
        min_radius=0.0,
        mul=50,
        layers=3,
        lmax=2,
        pool_nodes=True,
        number_of_basis=10,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.pool_nodes = pool_nodes

        irreps_node_hidden = o3.Irreps(
            [(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )

        self.mp = MessagePassing(
            irreps_node_input=irreps_node_input,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_node_output,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr
            + o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_node_input = self.mp.irreps_node_input
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_node_output = self.mp.irreps_node_output

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "batch" not in data:
            raise ValueError('"Please batch your data!"')

        batch = data["batch"]

        # The graph
        edge_src = data["edge_index"][1]
        edge_dst = data["edge_index"][0]

        # Edge attributes

        edge_vec = data["edge_vec"]
        edge_sh = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )
        edge_attr = torch.cat([data["edge_attr"], edge_sh], dim=1)

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            self.min_radius,
            self.max_radius,
            self.number_of_basis,
            basis="cosine",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        node_outputs = self.mp(
            data["node_input"],
            data["node_attr"],
            edge_src,
            edge_dst,
            edge_attr,
            edge_length_embedding,
        )

        if self.pool_nodes:
            return scatter(node_outputs, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return node_outputs


class Autoencoder(torch.nn.Module):
    """Autoencoder

    Example of usage:
     ```python

    input_dim = 784  # input dimensionality (28x28=784 for MNIST)
    latent_dim = 2   # latent space dimensionality
    encoder_layers = [128, 64, 12]  # encoder layer sizes
    decoder_layers = [12, 64, 128]  # decoder layer sizes

    # initialize the autoencoder model
    autoencoder = Autoencoder(input_dim, latent_dim, encoder_layers, decoder_layers)

    # define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

     ```
    """

    def __init__(self, input_dim, latent_dim, encoder_layers, decoder_layers):
        super(Autoencoder, self).__init__()

        # encoder network
        encoder = []
        prev_layer_dim = input_dim
        for layer_dim in encoder_layers:
            encoder.append(torch.nn.Linear(prev_layer_dim, layer_dim))
            encoder.append(torch.nn.ReLU())
            prev_layer_dim = layer_dim
        encoder.append(torch.nn.Linear(prev_layer_dim, latent_dim))
        self.encoder = torch.nn.Sequential(*encoder)

        # decoder network
        decoder = []
        prev_layer_dim = latent_dim
        for layer_dim in decoder_layers:
            decoder.append(torch.nn.Linear(prev_layer_dim, layer_dim))
            decoder.append(torch.nn.ReLU())
            prev_layer_dim = layer_dim
        decoder.append(torch.nn.Linear(prev_layer_dim, input_dim))
        # decoder.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*decoder)

    def forward(self, x):
        # encode the input
        encoded = self.encoder(x)

        # decode the encoded input
        decoded = self.decoder(encoded)

        return decoded
