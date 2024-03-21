
from torch_geometric.transforms import Compose

from .._src.frameworks_base import BaseChemicalMotifIdentifier
from .._src.transforms import AddEdges, ModuloConcentration, RemoveCentralNode
from .dataset import OvitoDataThermal


class BasePTMMotifIdentifier(BaseChemicalMotifIdentifier): 
    def __init__(self, rmsd_cutoff=0.1, **kwargs):
        self.mode = "synthetic"
        self.rmsd_cutoff = rmsd_cutoff
        super().__init__(**kwargs)

        dataset_config = {
            "pre_filter": None,
            "one_hot_dim": 5,
            "crystal_structure": self.crystal_structure,
            "rmsd_cutoff": self.rmsd_cutoff,
        }

        dataset_object = OvitoDataThermal
        self.set_dataset_modes(dataset_config, dataset_object)
        self.import_synthetic()

        self.nelement = 5
        self.transfroms = {
            "fcc": Compose(
                [
                    RemoveCentralNode(),
                    ModuloConcentration(self.nelement, self.num_nodes),
                ]
            ),
            "bcc": Compose(
                [
                    AddEdges(8),
                    RemoveCentralNode(),
                    ModuloConcentration(self.nelement, self.num_nodes),
                ]
            ),
            "hcp": Compose(
                [
                    RemoveCentralNode(),
                    ModuloConcentration(self.nelement, self.num_nodes),
                ]
            ),
        } 
    