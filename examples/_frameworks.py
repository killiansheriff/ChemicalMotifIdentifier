import numpy as np
import pandas as pd
from simplex import Simplex

from chemicalmotifidentifier import (
    BaseMonteCarloChemicalMotifIdentifier,
    BaseSyntheticChemicalMotifIdentifier,
)

# Model used in https://arxiv.org/abs/2311.01545

INPUT_GDOWN_LINK = 'https://drive.google.com/drive/folders/1VxK5mPu8bveaqFSSYnxrsKfZrTW_qXOX?usp=sharing' # Folder with model weights, sample graphs etc. Permission needs to be anyone with the link.

class SyntheticChemicalMotifIdentifier(BaseSyntheticChemicalMotifIdentifier):
    """Just a class that re use the framework above but that matches the parameters of the first paper.

    Args:
        ECA_Synthetic (_type_): _description_
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_config["one_hot_dim"] = 3

    def import_model_config(self):
        lmax, layers, outlength, number = 1, 1, 100, 0  # 2,2,4,0

        model_config = {
            "out_feature_length": outlength,
            "max_radius": 2.5,
            "min_radius": 0,
            "number_of_basis": 10,
            "num_nodes": 12,  # only used for feqtur enormalization, we don't really care set it to a constant so that we can have only 1 network for all the crystal structure
            "num_neighbors": 11,  # only used for feqtur enormalization, we don't really care set it to a constant so that we can have only 1 network for all the crystal structure
            "layers": layers,
            "lmax": lmax,
            "net_number": number,
            "irreps_node_attr": "3x0e",
            # "model_load": f"/home/ksheriff/PAPERS/first_paper/02_1nn_synthetic/data/nets/net_{lmax}-{layers}-{outlength}_{number}.pt",
            "model_load": "data/inputs_doi-10.48550-arXiv.2311.01545/net.pt",
            "mul": 50,  # 50
        }
        self.model_config = model_config

    def set_up_generator_space(self, yhat, concs, nelement=3):
        """Setup the generator space.

        Args:
            yhat (np.array): rounded fingerprint of microstate having generating concentrations.
            concs (np.array): concentrations associated with each yhat.
            nelement (int, optional): number of atomic type allowed in your sysnthetic dataset. Defaults to 3.

        Returns:
            tupe (generators, generator_space, inverses) : (concentration generators, physically constrained embeding space for these generators, inverses for each microstates / in wich concentration class they belong wrt to concentration. )
        """
        self.nelement = nelement
        concs = concs[:, : self.nelement]
        generators, inverses = np.unique(concs, axis=0, return_inverse=True)

        self.simplex = Simplex(
            n_dim=len(generators[0]) - 1, edge_length=1, nneigh=self.num_nodes
        )
        self.vertices = self.simplex.get_vertex_coordinates()

        # comvert bary_coords to cartesian
        cartesian_coords = self.simplex.barycenter_coods_2_cartesian(
            self.vertices, concs
        )

        # last dim is structural information

        structural_information = self.get_structural_information_PCA(
            yhat, inverses, rounding_number=8
        )

        generator_space = np.hstack(
            (cartesian_coords, structural_information.reshape(-1, 1))
        )

        return generators, generator_space, inverses


class MonteCarloChemicalMotifIdentifier(BaseMonteCarloChemicalMotifIdentifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nelement = 3
        self.dataset_config["one_hot_dim"] = 3

    def import_synthetic(self):
        """Import chemical shell synthetic dataset pandas dataframe"""
        self.df_synthetic = pd.read_pickle(
            f"data/inputs_doi-10.48550-arXiv.2311.01545/df_{self.crystal_structure}.pkl"
        )

    def import_model_config(self):
        lmax, layers, outlength, number = 1, 1, 100, 0  # 2,2,4,0

        model_config = {
            "out_feature_length": outlength,
            "max_radius": 2.5,
            "min_radius": 0,
            "number_of_basis": 10,
            "num_nodes": 12,  # only used for feqtur enormalization, we don't really care set it to a constant so that we can have only 1 network for all the crystal structure
            "num_neighbors": 11,  # only used for feqtur enormalization, we don't really care set it to a constant so that we can have only 1 network for all the crystal structure
            "layers": layers,
            "lmax": lmax,
            "net_number": number,
            "irreps_node_attr": "3x0e",
            # "model_load": f"/home/ksheriff/PAPERS/first_paper/02_1nn_synthetic/data/nets/net_{lmax}-{layers}-{outlength}_{number}.pt",
            "model_load": "data/inputs_doi-10.48550-arXiv.2311.01545/net.pt",
            "mul": 50,  # 50
        }
        self.model_config = model_config



    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def import_model_config(self):
        lmax, layers, outlength, number = 2, 2, 4, 0  # 2,2,4,0

        model_config = {
            "out_feature_length": outlength,
            "max_radius": 3,  # 2.5
            "min_radius": 0,
            "number_of_basis": 10,
            "num_nodes": 12,  # only used for feqtur enormalization, we don't really care set it to a constant so that we can have only 1 network for all the crystal structure
            "num_neighbors": 5,  # only used for feqtur enormalization, we don't really care set it to a constant so that we can have only 1 network for all the crystal structure
            "layers": layers,
            "lmax": lmax,
            "net_number": number,
            "irreps_node_attr": "5x0e",
            "model_load": f"/home/ksheriff/PAPERS/second_paper/02_1nn_synthetic/data/nets/net_{lmax}-{layers}-{outlength}_{number}.pt",
            "mul": 3,  # 50
        }
        self.model_config = model_config