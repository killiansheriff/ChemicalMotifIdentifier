import numpy as np
from ovito.data import CutoffNeighborFinder, DataCollection
from ovito.pipeline import ModifierInterface
from traits.api import Float, String

from .shell_information import NUMBER_OF_ATOMS_IN_SHELL


class NshellFinder(ModifierInterface):
    """ """

    cutoff = Float(default_value=18.2)
    crystal_structure = String(value="fcc")

    def get_cumsum_atom_in_shell(self):
        number_of_atoms_in_shells = NUMBER_OF_ATOMS_IN_SHELL[self.crystal_structure]

        cum_sum_atom_in_shell = np.zeros(len(number_of_atoms_in_shells) + 1)
        cum_sum_atom_in_shell[1:] = np.cumsum(number_of_atoms_in_shells)

        return cum_sum_atom_in_shell

    def get_nshell_neighbor_idx(self, nshell, cum_sum_atom_in_shell, neigh_idx):
        lb_count = int(cum_sum_atom_in_shell[nshell])
        up_count = int(cum_sum_atom_in_shell[nshell + 1])

        nn_ind_in_shell = neigh_idx[
            :,
            lb_count:up_count,
        ]
        return nn_ind_in_shell

    def modify(self, data: DataCollection, frame: int, **kwargs):
        finder = CutoffNeighborFinder(self.cutoff, data)
        neigh_idx, _ = finder.find_all(sort_by="distance")

        N = data.particles.count

        starts = np.searchsorted(neigh_idx[:, 0], np.arange(N), side="left")
        ends = np.searchsorted(neigh_idx[:, 0], np.arange(N), side="right")

        neigh_idx = np.array(
            [neigh_idx[starts[i] : ends[i]][:, 1] for i in range(N)]
        )  # (Natoms, Nneigh within cutoff)

        cum_sum_atom_in_shell = self.get_cumsum_atom_in_shell()

        max_shell_given_cutoff = np.argmax(
            np.where(cum_sum_atom_in_shell <= len(neigh_idx[0]))
        )

        neighbor_indices_per_shell = []

        for nshell in range(max_shell_given_cutoff):
            nshell_nn_indices = self.get_nshell_neighbor_idx(
                nshell, cum_sum_atom_in_shell, neigh_idx
            )
            neighbor_indices_per_shell.append(nshell_nn_indices)

        data.attributes["Neighbor indices per shell"] = neighbor_indices_per_shell
