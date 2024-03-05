import numpy as np
from tqdm import tqdm


class Dissimilarity:
    def __init__(self):
        pass

    @staticmethod
    def get_center_atom_dissim(neighbor_phys_emb, center_atom_phys_emb):
        center_neigh = neighbor_phys_emb[..., 0]
        centers = center_atom_phys_emb[..., None, 0]

        central_atom_dissim = (center_neigh != centers).astype(int)

        return central_atom_dissim

    @staticmethod
    def get_concentration_dissim(nprincipal, neighbor_phys_emb, center_atom_phys_emb):
        conc_cols = np.arange(
            1, nprincipal, 1
        )  # for nprincipal = 3, want [1,2] since in simplex space
        embeding_neigh = neighbor_phys_emb[..., conc_cols]
        embeding_center = center_atom_phys_emb[..., None, conc_cols]

        concentration_dissim = np.linalg.norm(embeding_center - embeding_neigh, axis=-1)

        return concentration_dissim

    @staticmethod
    def get_structural_dissim(nprincipal, neighbor_phys_emb, center_atom_phys_emb):
        embeding_neigh = neighbor_phys_emb[..., nprincipal:]
        embeding_center = center_atom_phys_emb[..., None, nprincipal:]

        structural_dissim = np.linalg.norm(embeding_center - embeding_neigh, axis=-1)

        return structural_dissim

    def get_separate_dissimilarities(
        self, nprincipal, neighbor_phys_emb, center_atom_phys_emb
    ):
        central_atom_dissim = self.get_center_atom_dissim(
            neighbor_phys_emb, center_atom_phys_emb
        )
        concentration_dissim = self.get_concentration_dissim(
            nprincipal, neighbor_phys_emb, center_atom_phys_emb
        )
        structural_dissim = self.get_structural_dissim(
            nprincipal, neighbor_phys_emb, center_atom_phys_emb
        )
        return central_atom_dissim, concentration_dissim, structural_dissim

    def get_dissimilarity_matrix(
        self, nprincipal, neighbor_phys_emb, center_atom_phys_emb
    ):
        """
        nprincipal: int
        neighbor_phys_emb: (Natoms, Nneigh, Nemb)
        center_atom_phys_emb (Natoms, Nemb)

        Returns D_per_atom_no_sum (Natoms, Nemb)
        """

        (
            central_atom_dissim,
            concentration_dissim,
            structural_dissim,
        ) = self.get_separate_dissimilarities(
            nprincipal, neighbor_phys_emb, center_atom_phys_emb
        )

        D_per_atom_no_sum = np.mean(
            [central_atom_dissim, concentration_dissim, structural_dissim], axis=2
        ).T  # (Natoms x 3)

        return D_per_atom_no_sum


class NshellDissimilarity:
    def __init__(self):
        pass

    @staticmethod
    def run(nn_idx, phys_emb, nprincipal):
        nshell_max = len(nn_idx)
        natoms = len(nn_idx[0])
        D = np.zeros((nshell_max, natoms, 3))  # (N_shells x Natoms x 3 dissim)

        for n in tqdm(range(nshell_max)):
            D_per_atom_no_sum = Dissimilarity().get_dissimilarity_matrix(
                nprincipal=nprincipal,
                neighbor_phys_emb=phys_emb[nn_idx[n]],
                center_atom_phys_emb=phys_emb,
            )
            D[n] = D_per_atom_no_sum
        return D


class FirstNShellDissim:
    def __init__(self, nshell_max, phys_emb, nn_idx, nprincipal):
        self.phys_emb = phys_emb
        self.nshell_max = nshell_max
        self.nprincipal = nprincipal

        self.nn_idx = []
        for n in range(self.nshell_max):
            self.nn_idx.append(nn_idx[n])

        self.get_counts()

    def get_counts(self):
        self.unique_phys_emb, inverse_indices, counts = np.unique(
            self.phys_emb,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        self.counts = counts[inverse_indices]
        self.inverse_indices = inverse_indices

        self.indices = [
            np.where(inverse_indices == self.inverse_indices[i])[0]
            for i in tqdm(range(len(self.phys_emb)), desc="masking")
        ]

    def list_of_neigh_for_similar_lcm(self, nshell, i):
        indices = self.indices[i]

        nn_ind_in_shell = self.nn_idx[nshell][indices]

        all_possible_nn_index_unique = np.unique(nn_ind_in_shell.flatten())

        _, index_of_unique_neigh = np.unique(
            self.inverse_indices[all_possible_nn_index_unique], return_index=True
        )

        all_possible_nn_index_unique_and_unique_motif = all_possible_nn_index_unique[
            index_of_unique_neigh
        ]

        weights = self.counts[all_possible_nn_index_unique_and_unique_motif]
        weights = weights / np.sum(weights)

        return all_possible_nn_index_unique_and_unique_motif, weights

    def get_dissim(self, nshell):
        N = len(self.phys_emb)
        D_per_atom_no_sum = np.zeros((N, 3))

        for i in range(N):
            nn, w = self.list_of_neigh_for_similar_lcm(nshell, i)
            nn_selected = nn

            neighbor_phys_emb = self.phys_emb[nn_selected]
            neighbor_phys_emb = neighbor_phys_emb.reshape(
                (1, neighbor_phys_emb.shape[0], neighbor_phys_emb.shape[1])
            )

            center_atom_phys_emb = np.array([self.phys_emb[i]])

            (
                central_atom_dissim,
                concentration_dissim,
                structural_dissim,
            ) = Dissimilarity().get_separate_dissimilarities(
                self.nprincipal, neighbor_phys_emb, center_atom_phys_emb
            )

            D_per_atom_no_sum_i = np.vstack(
                (
                    np.sum(central_atom_dissim * w, axis=1),
                    np.sum(concentration_dissim * w, axis=1),
                    np.sum(structural_dissim * w, axis=1),
                )
            ).T  # (1 x 3)

            D_per_atom_no_sum[i] = D_per_atom_no_sum_i[0]

        return D_per_atom_no_sum

    def run(self):
        D = np.zeros(
            (self.nshell_max, len(self.phys_emb), 3)
        )  # (N_shells x Natoms x 3 dissim)
        for nshell in tqdm(range(self.nshell_max), desc="Dissim per shell"):
            # (N, Possible index of 1nn minus the one in the current run)

            D_per_atom_no_sum = self.get_dissim(nshell)  # (Natoms, 3)

            D[nshell] = D_per_atom_no_sum

        return D


class BaselineDissim:
    def __init__(self, phys_emb, nprincipal):
        self.phys_emb = phys_emb
        self.nprincipal = nprincipal
        self.get_unique()

    def get_unique(self):
        self.unique_phys_emb, _, counts = np.unique(
            self.phys_emb,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        self.unique_weights = counts / np.sum(counts)

    def get_dissim(self):
        N = len(self.phys_emb)
        D_per_atom_no_sum = np.zeros((N, 3))

        for i in range(N):
            # select all unique ms

            center_atom_phys_emb = np.array([self.phys_emb[i]])

            neighbor_phys_emb = self.unique_phys_emb
            w = self.unique_weights

            neighbor_phys_emb = neighbor_phys_emb.reshape(
                (1, neighbor_phys_emb.shape[0], neighbor_phys_emb.shape[1])
            )

            (
                central_atom_dissim,
                concentration_dissim,
                structural_dissim,
            ) = Dissimilarity().get_separate_dissimilarities(
                self.nprincipal, neighbor_phys_emb, center_atom_phys_emb
            )

            D_per_atom_no_sum_i = np.vstack(
                (
                    np.sum(central_atom_dissim * w, axis=1),
                    np.sum(concentration_dissim * w, axis=1),
                    np.sum(structural_dissim * w, axis=1),
                )
            ).T  # (1 x 3)

            D_per_atom_no_sum[i] = D_per_atom_no_sum_i[0]

        return D_per_atom_no_sum

    def run(self):
        D = self.get_dissim()

        return D


class DissimilarityNoise(Dissimilarity):
    def __init__(
        self,
        phys_emb,
        nn_idx,
        nprincipal,
    ):
        self.phys_emb = phys_emb
        self.nprincipal = nprincipal
        self.natom_in_shell = []

        self.nshell_max = len(nn_idx)
        for n in range(self.nshell_max):
            self.natom_in_shell.append(len(nn_idx[n][0]))

    def run(self, nshell, N_stat):
        N_runs = int(
            N_stat / self.natom_in_shell[nshell]
        )  # number of runs requiered based on N_stats to get good statistics
        N_runs = np.max([N_runs, 200])
        N_runs = np.min([N_runs, 3000])

        D = np.zeros(
            (N_runs, self.phys_emb.shape[0], 3)
        )  # (N_runs x Natoms x 3 dissim)

        for i in tqdm(range(N_runs), desc="Run for nth shell"):
            nn_idx = np.random.randint(
                0,
                self.phys_emb.shape[0],
                size=(self.phys_emb.shape[0], int(self.natom_in_shell[nshell])),
            )
            D_per_atom_no_sum = Dissimilarity().get_dissimilarity_matrix(
                nprincipal=self.nprincipal,
                neighbor_phys_emb=self.phys_emb[nn_idx],
                center_atom_phys_emb=self.phys_emb,
            )

            D[i] = D_per_atom_no_sum

        return D
