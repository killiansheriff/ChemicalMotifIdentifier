import numpy as np
import pandas as pd
from simplex import Simplex

from chemicalmotifidentifier import (
    BaseMonteCarloChemicalMotifIdentifier,
    BaseSyntheticChemicalMotifIdentifier,
)

# Model used in LINK

INPUT_GDOWN_LINK = '' # Folder with model weights, sample graphs etc.

class SyntheticChemicalMotifIdentifier(BaseSyntheticChemicalMotifIdentifier):
    """Just a class that re use the framework above but that matches the parameters of the first paper.

    Args:
        ECA_Synthetic (_type_): _description_
    """

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
            "model_load": f"/home/ksheriff/PAPERS/second_paper/02_1nn_synthetic/data/nets/net_{lmax}-{layers}-{outlength}_{number}.pt", #!!!
            "mul": 3,  # 50
        }
        self.model_config = model_config

class MonteCarloChemicalMotifIdentifier(BaseMonteCarloChemicalMotifIdentifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def import_synthetic(self):
        """Import chemical shell synthetic dataset pandas dataframe"""
        self.df_synthetic = pd.read_pickle(
            f"/home/ksheriff/PAPERS/second_paper/02_1nn_synthetic/data/output/df_{self.crystal_structure}.pkl" # !!! 
        )
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
            "model_load": f"/home/ksheriff/PAPERS/second_paper/02_1nn_synthetic/data/nets/net_{lmax}-{layers}-{outlength}_{number}.pt", #!!! 
            "mul": 3,  # 50
        }
        self.model_config = model_config