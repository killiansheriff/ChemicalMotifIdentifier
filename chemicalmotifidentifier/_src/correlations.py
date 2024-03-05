import os
import pickle
from .shell_information import DISTANCE_OF_ATOMS_IN_SHELL
import numpy as np
from NshellFinder.concat import concat_nshell_neighbor_idx
from numpy.random import choice
from tqdm import tqdm

from .dissimilarity import (
    BaselineDissim,
    DissimilarityNoise,
    FirstNShellDissim,
    NshellDissimilarity,
)


class Correlations:
    @staticmethod
    def apply_weight_and_sum(D, weights):
        D = D * weights
        return D.sum(axis=-1)

    @staticmethod
    def get_graph_auto_correlations(D_sum, norm_factor=14):
        D_sum = D_sum.T
        phi = 1 - 2 * D_sum / norm_factor

        # Weighting by number of atoms in shell?
        return phi


class DomainSize:
    def __init__(
        self,
        dump_files,
        crystal_structure,
        lattice_parameter,
        nprincipal,
        root,
        cutoff,
        phys_emb,
        weights,
        norm_factor,
        N_stats,
    ):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.nprincipal = nprincipal
        self.crystal_structure = crystal_structure
        self.lattice_parameter = lattice_parameter
        self.dump_files = dump_files
        self.phys_emb = phys_emb
        self.weights = weights

        self.cutoff = cutoff
        self.norm_factor = norm_factor
        self.N_stats = N_stats

        n_baseline = {
            "fcc": 4,
            "bcc": 5,
            "hcp": 6,
        }  # pretty sure for bcc and fcc, hcp worked it out on paper because of floating point error in code to detect the overlaps
        self.n_baseline = n_baseline[self.crystal_structure]

    def get_nn_idx(self):
        file_path = self.root + "nn_idx.pkl"
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                self.nn_idx = pickle.load(file)
            return

        self.nn_idx = concat_nshell_neighbor_idx(
            self.dump_files,
            cutoff=self.cutoff,
            crystal_structure=self.crystal_structure,
        )
        with open(file_path, "wb") as file:
            pickle.dump(self.nn_idx, file)

    def get_nshell_max(self):
        self.nshell_max = len(self.nn_idx)

        self.natoms_in_shell = np.zeros(self.nshell_max)
        for n in range(self.nshell_max):
            self.natoms_in_shell[n] = len(self.nn_idx[n][0])
        np.save(self.root + "natoms_in_shells.npy", self.natoms_in_shell)

    def get_shell_distance(self):
        file_path = self.root + "shell_dst.npy"
        if os.path.exists(file_path):
            self.shell_dst = np.load(file_path)
            return

        file_shell = DISTANCE_OF_ATOMS_IN_SHELL[self.crystal_structure]
        self.shell_dst = file_shell[: self.nshell_max] * self.lattice_parameter
        np.save(file_path, self.shell_dst)

    def get_phi_infinity_first_n_shell(self):
        file_path = self.root + "phi_infinity_first_n_shell.npy"
        if os.path.exists(file_path):
            self.phi_infinity_first_n_shell = np.load(file_path)
            return

        D = FirstNShellDissim(
            nshell_max=self.n_baseline,
            phys_emb=self.phys_emb,
            nn_idx=self.nn_idx,
            nprincipal=self.nprincipal,
        ).run()
        np.save(self.root + "D_inf_first_n_shell.npy", D)

        D_sum = Correlations.apply_weight_and_sum(D, self.weights)

        self.phi_infinity_first_n_shell = Correlations.get_graph_auto_correlations(
            D_sum, self.norm_factor
        )
        np.save(
            file_path,
            self.phi_infinity_first_n_shell,
        )

    def get_phi_infinity_baseline(self):
        file_path = self.root + "phi_infinity_baseline.npy"
        if os.path.exists(file_path):
            self.phi_infinity_baseline = np.load(file_path)
            return

        D = BaselineDissim(phys_emb=self.phys_emb, nprincipal=self.nprincipal).run()
        np.save(self.root + "D_inf_baseline.npy", D)

        D_sum = Correlations.apply_weight_and_sum(D, self.weights)

        self.phi_infinity_baseline = Correlations.get_graph_auto_correlations(
            D_sum, self.norm_factor
        )
        np.save(
            self.root + "phi_infinity_baseline.npy",
            self.phi_infinity_baseline,
        )

    def get_phi_infinity(self):
        self.get_phi_infinity_first_n_shell()
        self.get_phi_infinity_baseline()

        self.phi_inf = np.zeros(
            (self.phi_infinity_first_n_shell.shape[0], len(self.shell_dst))
        )

        self.phi_inf[:, : self.n_baseline] = self.phi_infinity_first_n_shell
        self.phi_inf[:, self.n_baseline :] = self.phi_infinity_baseline.reshape((-1, 1))

        np.save(
            self.root + "phi_inf.npy",
            self.phi_inf,
        )

    def get_phi(self):
        file_path = self.root + "phi.npy"
        if os.path.exists(file_path):
            self.phi = np.load(file_path)
            return
        D = NshellDissimilarity.run(
            self.nn_idx, self.phys_emb, nprincipal=self.nprincipal
        )
        np.save(self.root + "D.npy", D)

        D_sum = Correlations.apply_weight_and_sum(D, self.weights)

        self.phi = Correlations.get_graph_auto_correlations(D_sum, self.norm_factor)

        np.save(self.root + "phi.npy", self.phi)
        np.save(self.root + "phi_avg.npy", self.phi.mean(axis=0))

    def get_C_0_abs_statistics(self, N_stat):
        if os.path.exists(self.root + "C_0_mean.npy") and os.path.exists(
            self.root + "C_0_std.npy"
        ):
            self.C_0_std = np.load(self.root + "C_0_std.npy")
            self.C_0_mean = np.load(self.root + "C_0_mean.npy")
            self.Ntrials_per_shell = np.load(self.root + "Ntrials_per_shell.npy")
            return

        nshell_max = len(self.nn_idx)
        natom_in_shell = []
        for n in range(self.nshell_max):
            natom_in_shell.append(len(self.nn_idx[n][0]))

        obj = DissimilarityNoise(
            phys_emb=self.phys_emb, nn_idx=self.nn_idx, nprincipal=self.nprincipal
        )

        N = self.phys_emb.shape[0]
        C_0_mean = np.zeros((nshell_max, N))
        C_0_std = np.zeros((nshell_max, N))
        Ntrials_per_shell = np.zeros(nshell_max)

        os.makedirs(self.root + "phi_noise/", exist_ok=True)
        for n in tqdm(range(nshell_max), desc="Dissim noise - shell"):
            if not os.path.exists(self.root + f"phi_noise_{n}.npy"):
                D_noise_n = obj.run(n, N_stat=N_stat)  # (Ntrials, Natoms, 3)
                D_noise_sum_n = Correlations.apply_weight_and_sum(
                    D=D_noise_n, weights=self.weights
                )
                phi_noise_n = Correlations.get_graph_auto_correlations(
                    D_sum=D_noise_sum_n, norm_factor=self.norm_factor
                )  # ( Natoms, Ntrials)
                np.save(self.root + f"phi_noise/phi_noise_{n}.npy", phi_noise_n)
            else:
                phi_noise_n = np.load(self.root + f"phi_noise_{n}.npy")

            C_0_n = phi_noise_n - self.phi_infinity_baseline.reshape(-1, 1)

            # C_0_abs_n = np.abs(C_0)

            C_0_n_mean = np.mean(C_0_n, axis=1)

            Ntrials_per_shell[n] = C_0_n.shape[1]
            C_0_n_std = np.std(C_0_n, ddof=1, axis=1)

            C_0_mean[n] = C_0_n_mean
            C_0_std[n] = C_0_n_std

        self.C_0_mean = C_0_mean.T  # (Natoms, Nshell)
        self.C_0_std = C_0_std.T  # (Natoms, Nshell)
        self.Ntrials_per_shell = Ntrials_per_shell

        np.save(self.root + "C_0_mean.npy", self.C_0_mean)
        np.save(self.root + "C_0_std.npy", self.C_0_std)
        np.save(self.root + "Ntrials_per_shell.npy", Ntrials_per_shell)

    def get_C(self):
        self.C = self.phi - self.phi_inf
        C_avg = self.C.mean(axis=0)
        np.save(self.root + "C.npy", self.C)
        np.save(self.root + "C_avg.npy", C_avg)

    def get_estimate_C_0_cumsum_std(self):
        C_0_abs_area_mean = np.cumsum(self.C_0_abs_mean, axis=1)
        np.save(self.root + "C_0_abs_area_mean.npy", C_0_abs_area_mean)

        epsilon_left = np.sqrt(np.cumsum((self.C_0_abs_std) ** 2, axis=1))

        np.save(self.root + "epsilon_left.npy", epsilon_left)

        # Gettin epsilon right but without considering the std of the current point
        epsilon_right = np.zeros(epsilon_left.shape)
        epsilon_right_tmp = np.sqrt(np.cumsum((self.C_0_abs_std[:, ::-1]) ** 2, axis=1))
        epsilon_right_tmp = epsilon_right_tmp[:, ::-1]
        epsilon_right_tmp = epsilon_right_tmp[:, 1:]
        epsilon_right[:, 0:-1] = epsilon_right_tmp
        np.save(self.root + "epsilon_right.npy", epsilon_right)

        self.epsilon_right = epsilon_right
        self.epsilon_left = epsilon_left
        self.C_0_abs_area_mean = C_0_abs_area_mean

    def get_C_area(self):
        self.C_area = np.cumsum(np.abs(self.C), axis=1)
        np.save(self.root + "C_area.npy", self.C_area)

    def get_sliding_window(self, arr_size: int, window_size: int):
        if window_size % 2 == 0:
            raise ValueError("window_size cannot be divisible by 2.")
        windows = []
        for i in range(arr_size):
            if i >= window_size // 2 and i < arr_size - window_size // 2:
                windows.append(
                    np.arange(i - window_size // 2, i + window_size // 2 + 1, 1)
                )
            elif i < window_size // 2:
                windows.append(np.arange(0, i + window_size // 2 + 1))
            elif i >= arr_size - window_size // 2:
                # windows.append(np.arange(arr_size - window_size, arr_size))

                windows.append(np.arange(i - window_size // 2, arr_size))

        assert arr_size == len(windows)

        self.windows = windows

    def get_C_moving(self):
        nshell = len(self.windows)
        self.C_moving = np.zeros(self.C.shape)
        for n in range(nshell):
            data = self.C[:, self.windows[n]] * self.natoms_in_shell[self.windows[n]]
            self.C_moving[:, n] = np.sum(data, axis=1) / np.sum(
                self.natoms_in_shell[self.windows[n]]
            )
        np.save(self.root + "C_moving.npy", self.C_moving)

    def get_C_0_mean_moving(self):
        nshell = len(self.windows)
        self.C_0_mean_moving = np.zeros(self.C_0_mean.shape)

        for n in range(nshell):
            data = (
                self.C_0_mean[:, self.windows[n]]
                * self.natoms_in_shell[self.windows[n]]
            )
            self.C_0_mean_moving[:, n] = np.sum(data, axis=1) / np.sum(
                self.natoms_in_shell[self.windows[n]]
            )

        np.save(self.root + "C_0_mean_moving.npy", self.C_0_mean_moving)

    def get_C_0_std_moving(self):
        nshell = len(self.windows)
        self.C_0_std_moving = np.zeros(self.C_0_std.shape)
        self.C_0_std_moving_err = np.zeros(self.C_0_std.shape)

        for n in range(nshell):
            data = (
                self.C_0_std[:, self.windows[n]] * self.natoms_in_shell[self.windows[n]]
            )
            self.C_0_std_moving[:, n] = np.sqrt(np.sum(data**2, axis=1)) / np.sum(
                self.natoms_in_shell[self.windows[n]]
            )

            self.C_0_std_moving_err[:, n] = np.sqrt(
                np.sum(
                    (data / np.sqrt(self.Ntrials_per_shell[self.windows[n]])) ** 2,
                    axis=1,
                )
            ) / np.sum(self.natoms_in_shell[self.windows[n]])
        np.save(self.root + "C_0_std_moving.npy", self.C_0_std_moving)
        np.save(self.root + "C_0_std_moving_err.npy", self.C_0_std_moving_err)

    def get_length_scale_distribution_from_moving_averages_with_curve_fit(
        self, std_factor
    ):
        N = self.C_moving.shape[0]
        nshell = self.C_moving.shape[1]
        ls = np.zeros(N)
        r = self.shell_dst

        # Calculate is_within_bounds for all N values at once

        is_within_bounds = np.abs(self.C_moving) <= std_factor * self.C_0_std_moving

        # Calculate the condition where all values are within bounds
        all_within_condition = np.all(is_within_bounds, axis=1)

        # Calculate the condition where all values are outside bounds
        all_outside_condition = np.all(~is_within_bounds, axis=1)

        # Initialize ls
        ls = np.ones(shape=(nshell, N)) * np.nan

        for shell_idx in range(nshell):
            is_within_window = is_within_bounds[:, self.windows[shell_idx]]

            # The ones below shell_idx should be signal, the ones after should be noise
            pattern = np.ones_like(is_within_window, dtype=bool)
            pattern[:, self.windows[shell_idx] <= shell_idx] = False

            within_shell_condition = np.all(pattern == is_within_window, axis=1)
            within_shell_condition &= np.all(
                ~is_within_bounds[:, : shell_idx + 1], axis=1
            )

            ls[shell_idx][within_shell_condition] = 2 * r[shell_idx]

        ls = ls.T  # natoms x nshells
        ls[np.isnan(ls)] = 0  # I think this is useless np.isnan(ls).sum() always == 0

        ls = np.nanmax(ls, axis=1)
        # Set ls values based on conditions
        ls[all_within_condition] = 0
        ls[all_outside_condition] = 2 * r[-1]

        # rs = np.linspace(0, self.shell_dst, 100)
        intercepts = np.zeros(N)
        fit_params = np.zeros((N, 3))

        for i in np.where(ls != 0)[0]:
            C_mov_i = self.C_moving[i]
            idx = int(np.where(self.shell_dst * 2 == ls[i])[0])

            shift = 3

            # lb = np.min(np.abs([idx - shift, 0]))
            lb = np.max([idx - shift, 0])
            ub = idx + shift

            m, b = np.polyfit(
                2 * self.shell_dst[lb:ub],
                C_mov_i[lb:ub],
                1,
            )
            fit_params[i] = np.array([m, b, 2 * self.shell_dst[idx]])
            intercept = -b / m
            intercepts[i] = intercept

            closest = 2 * r[np.min([np.argmin(np.abs(2 * r - intercept)), len(r) - 1])]

            ls[i] = closest

        self.L = ls
        self.ls_mask = np.logical_and(
            self.L > 0, self.L < 35
        )  #!!! HARDCODED VALUE AT 35A HERE
        self.L_mean = np.mean(self.L[self.ls_mask])
        self.L_std = np.std(self.L[self.ls_mask], ddof=1)
        self.L_count_without_zero = len(self.L[self.ls_mask])

        np.save(self.root + f"L_{std_factor}sigma.npy", self.L)
        np.save(self.root + f"L_mean_{std_factor}sigma.npy", self.L_mean)
        np.save(self.root + f"L_std_{std_factor}sigma.npy", self.L_std)
        np.save(self.root + f"L_mask_{std_factor}sigma.npy", self.ls_mask)

        np.save(
            self.root + f"L_count_without_zero_{std_factor}sigma.npy",
            self.L_count_without_zero,
        )
        print(
            f"{std_factor=} {self.L_mean=} {self.L_std=} {self.L_count_without_zero=}"
        )

        # Deal with intercepts
        self.intercepts = intercepts
        np.save(self.root + f"intercepts_{std_factor}sigma.npy", self.intercepts)
        np.save(self.root + f"fit_params_{std_factor}sigma.npy", fit_params)

        self.inter_mask = np.logical_and(
            self.intercepts > 0, self.intercepts < 35
        )  #!!! HARDCODED VALUE AT 35A HERE and at FIRST SHELL

        L_inter_mean = np.mean(self.intercepts[self.inter_mask])
        L_inter_std = np.std(self.intercepts[self.inter_mask], ddof=1)

        np.save(self.root + f"L_intercepts_mean_{std_factor}sigma.npy", L_inter_mean)
        np.save(self.root + f"L_intercepts_std_{std_factor}sigma.npy", L_inter_std)

        L_hist_count_intercepts, L_hist_bins_intercepts = np.histogram(
            self.intercepts[
                np.logical_and(self.intercepts >= 0, self.intercepts < 35)
            ],  #!!! HARDCODED VALUE AT 35A HERE
            500,
            density=False,
        )

        L_hist_bins_intercepts = L_hist_bins_intercepts[:-1]

        np.save(
            self.root + f"L_hist_count_intercepts_{std_factor}sigma.npy",
            L_hist_count_intercepts,
        )
        np.save(
            self.root + f"L_hist_bins_intercepts_{std_factor}sigma.npy",
            L_hist_bins_intercepts,
        )

    def get_length_scale_histogram(self, std_factor):
        self.L_hist_bins = np.zeros(len(self.shell_dst) + 1)
        self.L_hist_bins[1:] = self.shell_dst * 2

        histogram = dict(zip(self.L_hist_bins, np.zeros(self.L_hist_bins.shape)))

        L_unique, count = np.unique(self.L, return_counts=True)

        for i, l in enumerate(L_unique):
            histogram[l] = count[i]

        self.L_hist_count = np.array(list(histogram.values()))

        np.save(self.root + f"L_hist_count_{std_factor}sigma.npy", self.L_hist_count)
        np.save(self.root + f"L_hist_bins_{std_factor}sigma.npy", self.L_hist_bins)

    @staticmethod
    def get_length_scale_error(L, M):
        means = np.zeros(M)
        for m in range(M):
            new_L = choice(L, len(L), replace=True)
            means[m] = np.mean(new_L)

        # self.L_err = np.std(means, axis=0, ddof=1)

        means = np.sort(means)
        means = means[int(0.025 * M) : int(0.975 * M)]
        L_err = (means.max() - means.min()) / 2
        return L_err

    def main(self):
        print("Get the NNs")
        self.get_nn_idx()

        self.get_nshell_max()
        self.get_shell_distance()

        print("Get the phis")
        self.get_phi_infinity()
        self.get_phi()

        print("Get the Cs")
        self.get_C()
        self.get_C_0_abs_statistics(N_stat=self.N_stats)

        print("Get the Areas")
        WINDOW_SIZE = 7
        self.get_sliding_window(
            arr_size=self.shell_dst.shape[0], window_size=WINDOW_SIZE
        )
        self.get_C_moving()
        self.get_C_0_mean_moving()
        self.get_C_0_std_moving()

        # self.get_estimate_C_0_cumsum_std()
        # self.get_C_area()

        print("Get the distributions")
        for std_factor in [1, 2, 3, 4, 5]:
            # self.get_length_scale_distribution_from_moving_averages(std_factor=std_factor)
            # self.get_length_scale_distribution_conf_interval(std_factor=std_factor)
            self.get_length_scale_distribution_from_moving_averages_with_curve_fit(
                std_factor=std_factor
            )

            self.get_length_scale_histogram(std_factor=std_factor)

            M = 200  # bootstarp number
            L_err = self.get_length_scale_error(self.L[self.ls_mask], M)
            np.save(self.root + f"L_err_{std_factor}sigma.npy", L_err)

            L_inter_err = self.get_length_scale_error(
                self.intercepts[self.inter_mask], M
            )
            np.save(self.root + f"L_intercepts_err_{std_factor}sigma.npy", L_inter_err)

    @classmethod
    def reload(cls, root):
        obj = cls(
            dump_files="",
            crystal_structure="fcc",
            lattice_parameter="",
            nprincipal="",
            root=root,
            cutoff="",
            phys_emb="",
            weights="",
            norm_factor="",
            N_stats="",
        )

        obj.shell_dst = np.load(root + "shell_dst.npy")
        obj.C = np.load(root + "C.npy")
        obj.C_0_mean = np.load(root + "C_0_mean.npy")
        obj.C_0_std = np.load(root + "C_0_std.npy")

        obj.L_hist_bins_3sigma = np.load(root + "L_hist_bins_3sigma.npy")
        obj.L_hist_count_3sigma = np.load(root + "L_hist_count_3sigma.npy")
        obj.L_3sigma = np.load(root + "L_3sigma.npy")

        obj.L_hist_bins_2sigma = np.load(root + "L_hist_bins_2sigma.npy")
        obj.L_hist_count_2sigma = np.load(root + "L_hist_count_2sigma.npy")
        obj.L_2sigma = np.load(root + "L_2sigma.npy")

        obj.L_hist_bins_1sigma = np.load(root + "L_hist_bins_1sigma.npy")
        obj.L_hist_count_1sigma = np.load(root + "L_hist_count_1sigma.npy")
        obj.L_1sigma = np.load(root + "L_1sigma.npy")

        obj.natoms_in_shell = np.load(root + "natoms_in_shells.npy")

        obj.C_moving = np.load(root + "C_moving.npy")
        obj.C_0_mean_moving = np.load(root + "C_0_mean_moving.npy")
        obj.C_0_std_moving = np.load(root + "C_0_std_moving.npy")

        obj.fit_apram_2sigma = np.load(root + "fit_params_2sigma.npy")
        obj.intercepts_2sigma = np.load(root + "intercepts_2sigma.npy")

        return obj
