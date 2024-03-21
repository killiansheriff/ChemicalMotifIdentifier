import numpy as np
import torch
from ovito.data import NearestNeighborFinder
from ovito.io import import_file
from ovito.modifiers import (
    CalculateDisplacementsModifier,
    ExpressionSelectionModifier,
    PolyhedralTemplateMatchingModifier,
)
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

######################
# Ovito data defect
######################


ROOT_2_OVER_2 = np.sqrt(2) / 2
FCC = [
    [ROOT_2_OVER_2, ROOT_2_OVER_2, 0],
    [0, ROOT_2_OVER_2, ROOT_2_OVER_2],
    [ROOT_2_OVER_2, 0, ROOT_2_OVER_2],
    [-ROOT_2_OVER_2, -ROOT_2_OVER_2, 0],
    [0, -ROOT_2_OVER_2, -ROOT_2_OVER_2],
    [-ROOT_2_OVER_2, 0, -ROOT_2_OVER_2],
    [-ROOT_2_OVER_2, ROOT_2_OVER_2, 0],
    [0, -ROOT_2_OVER_2, ROOT_2_OVER_2],
    [-ROOT_2_OVER_2, 0, ROOT_2_OVER_2],
    [ROOT_2_OVER_2, -ROOT_2_OVER_2, 0],
    [0, ROOT_2_OVER_2, -ROOT_2_OVER_2],
    [ROOT_2_OVER_2, 0, -ROOT_2_OVER_2],
]
ROOT_3_OVER_2 = np.sqrt(3) / 2
ROOT_6_OVER_3 = np.sqrt(6) / 3
ROOT_3_OVER_6 = np.sqrt(3) / 6
ROOT_3_OVER_3 = np.sqrt(3) / 3
HCP1 = [
    [0.5, -ROOT_3_OVER_2, 0],
    [-1, 0, 0],
    [-0.5, ROOT_3_OVER_6, -ROOT_6_OVER_3],
    [0.5, ROOT_3_OVER_6, -ROOT_6_OVER_3],
    [0, -ROOT_3_OVER_3, -ROOT_6_OVER_3],
    [-0.5, ROOT_3_OVER_2, 0],
    [0.5, ROOT_3_OVER_2, 0],
    [1, 0, 0],
    [-0.5, -ROOT_3_OVER_2, 0],
    [0, -ROOT_3_OVER_3, ROOT_6_OVER_3],
    [0.5, ROOT_3_OVER_6, ROOT_6_OVER_3],
    [-0.5, ROOT_3_OVER_6, ROOT_6_OVER_3],
]
HCP2 = [
    [1, 0, 0],
    [-0.5, -ROOT_3_OVER_2, 0],
    [-0.5, -ROOT_3_OVER_6, -ROOT_6_OVER_3],
    [0, ROOT_3_OVER_3, -ROOT_6_OVER_3],
    [0.5, -ROOT_3_OVER_6, -ROOT_6_OVER_3],
    [-1, 0, 0],
    [-0.5, ROOT_3_OVER_2, 0],
    [0.5, ROOT_3_OVER_2, 0],
    [0.5, -ROOT_3_OVER_2, 0],
    [0.5, -ROOT_3_OVER_6, ROOT_6_OVER_3],
    [0, ROOT_3_OVER_3, ROOT_6_OVER_3],
    [-0.5, -ROOT_3_OVER_6, ROOT_6_OVER_3],
]


def get_edges(vertexpositions, nn_dst, atol=0.1):
    # Subtract each point from all the other points
    diff = vertexpositions[:, np.newaxis, :] - vertexpositions[np.newaxis, :, :]

    # Compute the norm of the differences along the last axis
    distances = np.linalg.norm(diff, axis=-1)

    mask_1nn_2_1nn = np.isclose(distances, nn_dst, atol=atol)
    edge_index_0, edge_index_1 = np.where(mask_1nn_2_1nn)

    return edge_index_0, edge_index_1, diff[mask_1nn_2_1nn]


def get_matrix(q):
    q = np.array(q)
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    matrices = np.empty((3, 3))
    matrices[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    matrices[0, 1] = 2.0 * (qx * qy - qw * qz)
    matrices[0, 2] = 2.0 * (qx * qz + qw * qy)
    matrices[1, 0] = 2.0 * (qx * qy + qw * qz)
    matrices[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    matrices[1, 2] = 2.0 * (qy * qz - qw * qx)
    matrices[2, 0] = 2.0 * (qx * qz - qw * qy)
    matrices[2, 1] = 2.0 * (qy * qz + qw * qx)
    matrices[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)

    return matrices


class OvitoDataThermal(InMemoryDataset):
    def __init__(
        self,
        root,
        dump_file,
        nneigh,
        frame_number,
        rounding_edge_vec_value,
        crystal_structure,
        rmsd_cutoff=0.1,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        one_hot_dim=5,
    ):
        self.nneigh = nneigh
        self.rmsd_cutoff = rmsd_cutoff
        self.dump_file = dump_file
        self.frame_number = frame_number
        self.rounding_edge_vec_value = rounding_edge_vec_value
        self.one_hot_dim = one_hot_dim
        self.crystal_structure = crystal_structure

        structure_2_PTM = {"fcc": 1, "hcp": 2, "bcc": 3}
        self.crystal_structure = crystal_structure
        self.StructureType = structure_2_PTM[self.crystal_structure]

        # reference_structure = {"fcc": FCC, "hcp": HCP2}
        # self.reference_structure_vecs = reference_structure[self.crystal_structure]

        nn_2_nn_dst_crystals = {"fcc": 1, "hcp": 1}
        self.nn_2_nn_dst_crystals = nn_2_nn_dst_crystals[self.crystal_structure]

        remap_functions = {"fcc": self.remap_fcc_nn_vecs, "hcp": self.remap_hcp_nn_vecs}
        self.remap_function = remap_functions[self.crystal_structure]

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        (The download func. is not implemented here)
        """
        return ["not_implemented.pt"]

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def get_per_atom_energies(self, pipeline):
        try:
            data_relaxed = pipeline.compute(1)
            c_peratom = np.array(data_relaxed.particles["c_peratom"])
            return c_peratom
            # os.makedirs(self.root + "/per_atom_quantity/", exist_ok=True)
            # torch.save(c_peratom, self.root + "/per_atom_quantity/energy.pt")
        except:
            print("No c_peratom quantity")

    def get_per_atom_displacements(self, pipeline):
        try:
            pipeline.modifiers.append(CalculateDisplacementsModifier())
            data_relaxed = pipeline.compute(1)
            disp = np.array(data_relaxed.particles["Displacement Magnitude"])
            return disp
            # os.makedirs(self.root + "/per_atom_quantity/", exist_ok=True)
            # torch.save(disp, self.root + "/per_atom_quantity/disp.pt")
        except:
            print("No relaxed quantity")

    def get_pipeline_and_data(self):
        # Overwirtting get pipeline to add selection modifiers to only select perfect structures
        pipeline = import_file(self.dump_file)
        print(self.dump_file)

        # PTM and selection modifier

        ptm = PolyhedralTemplateMatchingModifier()
        ptm.only_selected = False
        ptm.rmsd_cutoff = self.rmsd_cutoff
        ptm.output_interatomic_distance = True
        ptm.output_orientation = True
        ptm.output_ordering = True

        pipeline.modifiers.append(ptm)
        pipeline.modifiers.append(
            ExpressionSelectionModifier(
                expression=f"StructureType =={self.StructureType}"
            )
        )

        data = pipeline.compute(int(self.frame_number))

        return pipeline, data

    @staticmethod
    def get_angles(nn_vecs, reference_vecs):
        dot_products = np.dot(nn_vecs, reference_vecs.T)

        vectors_magnitudes = np.linalg.norm(nn_vecs, axis=-1, keepdims=True)
        fcc_magnitudes = np.linalg.norm(reference_vecs, axis=-1)

        magnitude_products = vectors_magnitudes * fcc_magnitudes

        cosine_thetas = dot_products / magnitude_products
        cosine_thetas = np.clip(cosine_thetas, -1.0, 1.0)

        angles = np.arccos(cosine_thetas)
        return angles

    def remap_hcp_nn_vecs(self, data, nn_vecs):
        orientations = data.particles["Orientation"].array
        structure_types = data.particles["Structure Type"].array

        rotation_matrices = np.array([get_matrix(q) for q in orientations])
        nn_vecs = np.einsum("aij,ajk->aik", nn_vecs, rotation_matrices)

        reference_vecs1 = np.array(HCP1)
        reference_vecs2 = np.array(HCP2)

        angles_HCP1 = self.get_angles(nn_vecs, reference_vecs1)
        angles_HCP2 = self.get_angles(nn_vecs, reference_vecs2)

        best_matches_HCP1 = np.argmin(angles_HCP1, axis=-1)
        best_matches_HCP2 = np.argmin(angles_HCP2, axis=-1)

        avg_angle_diff_HCP1 = np.mean(np.min(angles_HCP1, axis=-1), axis=-1)
        avg_angle_diff_HCP2 = np.mean(np.min(angles_HCP2, axis=-1), axis=-1)

        mapped_nn_vecs = reference_vecs2[best_matches_HCP2]

        best_matches = best_matches_HCP2
        mask = np.where(avg_angle_diff_HCP1 < avg_angle_diff_HCP2)[0]

        mapped_nn_vecs[mask] = reference_vecs1[best_matches_HCP1][mask]

        best_matches[mask] = best_matches_HCP1[mask]

        # masking out the ones that have more than 1 time a the same vecs
        all_edge_different = np.cumsum(np.arange(self.nneigh))[-1]

        selected_particles = np.logical_and(
            np.sum(best_matches, axis=-1) == all_edge_different,
            np.array(structure_types) == self.StructureType,
        )

        best_match_selected = best_matches[selected_particles]

        assert len(np.unique(np.sum(best_match_selected, axis=-1))) == 1
        return mapped_nn_vecs, np.where(selected_particles == True)[0]

    def remap_fcc_nn_vecs(self, data, nn_vecs):
        orientations = data.particles["Orientation"].array
        structure_types = data.particles["Structure Type"].array

        rotation_matrices = np.array([get_matrix(q) for q in orientations])
        nn_vecs = np.einsum("aij,ajk->aik", nn_vecs, rotation_matrices)

        reference_vecs = np.array(FCC)
        # Computing angle
        angles = self.get_angles(nn_vecs, reference_vecs)
        best_matches = np.argmin(angles, axis=-1)

        nn_vecs_match = reference_vecs[best_matches]

        # masking out the ones that have more than 1 time a the same vecs
        all_edge_different = np.cumsum(np.arange(self.nneigh))[-1]

        selected_particles = np.logical_and(
            np.sum(best_matches, axis=-1) == all_edge_different,
            np.array(structure_types) == self.StructureType,
        )

        best_match_selected = best_matches[selected_particles]

        assert len(np.unique(np.sum(best_match_selected, axis=-1))) == 1

        return nn_vecs_match, np.where(selected_particles == True)[0]

    def process(self):
        # Read data into huge `Data` list.

        pipeline, data = self.get_pipeline_and_data()

        # get per atom properties
        energy = self.get_per_atom_energies(pipeline)
        disp = self.get_per_atom_displacements(pipeline)

        # get atom types
        atom_types = np.array(data.particles.particle_types)

        natoms = data.particles.count

        # setup onehot encoder of atom types
        atom_types_one_hot = torch.eye(self.one_hot_dim, dtype=torch.float64)[
            atom_types - 1
        ]

        # setup NN finder
        finder = NearestNeighborFinder(self.nneigh, data)
        nn_indexs, nn_vecs = finder.find_all()

        nn_vecs, selected_particles = self.remap_function(data, nn_vecs)

        atom_indices = torch.arange(natoms)
        # create neighborhood matrix
        neighborhood = torch.zeros((natoms, self.nneigh + 1), dtype=int)
        neighborhood[:, 0] = atom_indices
        neighborhood[:, 1:] = torch.from_numpy(nn_indexs)

        data_list = []

        # loop over central atom i
        for iatom in tqdm(range(natoms), desc="Dataset atom i..."):
            # central atom i at index 0 with its NN
            ineighborhood = neighborhood[iatom]

            # get edge index for 1nn-1nn edges, shift them by one since first atom is center
            edge_index_0, edge_index_1, edge_vec = get_edges(
                nn_vecs[iatom], nn_dst=self.nn_2_nn_dst_crystals, atol=0.1
            )
            edge_index_0 = edge_index_0 + 1
            edge_index_1 = edge_index_1 + 1

            # Add central atom to 1nn information

            edge_index_0 = np.concatenate(
                (
                    [0] * self.nneigh,
                    np.arange(1, self.nneigh + 1),
                    edge_index_0,
                )
            )
            edge_index_1 = np.concatenate(
                (
                    np.arange(1, self.nneigh + 1),
                    [0] * self.nneigh,
                    edge_index_1,
                )
            )

            edge_vec = np.concatenate((nn_vecs[iatom], nn_vecs[iatom], edge_vec))

            edge_index_0 = torch.from_numpy(edge_index_0)
            edge_index_1 = torch.from_numpy(edge_index_1)
            edge_vec = torch.from_numpy(edge_vec)

            edge_index = torch.stack((edge_index_0, edge_index_1))

            edge_vec = torch.tensor(edge_vec, dtype=torch.float64)

            node_attr = atom_types_one_hot[ineighborhood]

            num_edges = edge_index.shape[1]
            node_input = torch.ones(self.nneigh + 1, 1, dtype=torch.double)
            edge_attr = torch.ones(num_edges, 1, dtype=torch.double)

            g = {
                "edge_index": edge_index,
                "edge_vec": edge_vec,
                "node_attr": node_attr,
                "index": iatom,
                "edge_attr": edge_attr,
                "node_input": node_input,
            }

            if disp is not None:
                g["disp"] = disp[iatom]

            if energy is not None:
                g["energy"] = energy[iatom]

            graph = Data(**g)
            data_list.append(graph)

        if selected_particles is not None:
            # selected_particles = np.where(np.array(selected_mask) == 1)[0]

            data_list = [data_list[i] for i in selected_particles]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
