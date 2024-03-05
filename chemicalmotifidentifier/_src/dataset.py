import numpy as np
import torch
from ovito.data import NearestNeighborFinder
from ovito.io import import_file
from ovito.modifiers import CalculateDisplacementsModifier
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class OvitoData(InMemoryDataset):
    def __init__(
        self,
        root,
        dump_file,
        nneigh,
        frame_number,
        rounding_edge_vec_value,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        one_hot_dim=5,
    ):
        self.nneigh = nneigh
        self.dump_file = dump_file
        self.frame_number = frame_number
        self.rounding_edge_vec_value = rounding_edge_vec_value
        self.one_hot_dim = one_hot_dim
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
        # read lammps file in ovito
        pipeline = import_file(self.dump_file)
        data = pipeline.compute(int(self.frame_number))
        return pipeline, data

    def map_to_perfect_disp(self, edge_vec):
        atol = 0.01
        possible_displacements = [
            0,
            np.sqrt(2) / 2,
            np.sqrt(3) / 3,
            np.sqrt(3) / 2,
            1.0000,
            np.sqrt(3) / 6,
            np.sqrt(2 / 3),
            0.5,
        ]

        for possible_displacement in possible_displacements:
            map_possible_displacement = possible_displacement

            edge_vec[
                torch.isclose(
                    edge_vec,
                    torch.tensor(possible_displacement, dtype=torch.float64),
                    atol=atol,
                )
            ] = map_possible_displacement
            edge_vec[
                torch.isclose(
                    edge_vec,
                    torch.tensor(-possible_displacement, dtype=torch.float64),
                    atol=atol,
                )
            ] = -map_possible_displacement
        return edge_vec

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
        # np.save(self.root + "nn_vecs.npy", nn_vecs)
        # np.save(self.root + "nn_idx.npy", nn_indexs)

        atom_indices = torch.arange(natoms)
        # create neighborhood matrix
        neighborhood = torch.zeros((natoms, self.nneigh + 1), dtype=int)
        neighborhood[:, 0] = atom_indices
        neighborhood[:, 1:] = torch.from_numpy(nn_indexs)

        data_list = []

        # loop over central atom i
        for iatom in tqdm(range(natoms)):
            # central atom i at index 0 with its NN
            ineighborhood = neighborhood[iatom]

            # if neighborhoods of each atom share also has a boudn with a 1nn
            neighborhoods = neighborhood[ineighborhood][:, 1:]

            common_neighbor = np.isin(
                neighborhoods,
                ineighborhood,
            )
            # common_neighbor[neighborhoods==iatom]=False

            idx_0, idx_1 = np.where(common_neighbor)

            edges = torch.tensor(
                [(ineighborhood[i], neighborhoods[i, j]) for i, j in zip(idx_0, idx_1)],
                dtype=torch.long,
            )

            # Use tensor slicing to extract edge indices
            edge_index_0, edge_index_1 = edges[:, 0], edges[:, 1]

            # Create a tensor for the mapping function using gather function
            ovito_idx_to_idx = torch.zeros(
                ineighborhood.max() + 1, dtype=torch.long, device=edges.device
            )
            ovito_idx_to_idx[ineighborhood] = torch.arange(
                len(ineighborhood), device=edges.device
            )
            map_func = lambda x: ovito_idx_to_idx.gather(0, x)

            # Apply the mapping function inplace using tensor indexing
            edge_index_0 = map_func(
                edge_index_0,
            )
            edge_index_1 = map_func(
                edge_index_1,
            )

            # Stack the tensors into a tensor
            edge_index = torch.stack((edge_index_0, edge_index_1))

            edge_vec = np.array(
                [nn_vecs[ineighborhood[i]][j] for i, j in zip(idx_0, idx_1)]
            )

            # removing lattice parameter effect
            edge_vec = torch.tensor(edge_vec, dtype=torch.float64)

            avg_NNdist = torch.mean(
                edge_vec[: self.nneigh].norm(dim=1), dtype=torch.float64
            )
            edge_vec = edge_vec / (avg_NNdist)

            # mapping to the perfect microstate skeleton

            if self.rounding_edge_vec_value:
                edge_vec = self.map_to_perfect_disp(edge_vec)

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

        # dealing with selected modifiers if it exists; could be optimize not to compute graph for all

        selected_mask = data.particles.selection

        if selected_mask is not None:
            selected_particles = np.where(np.array(selected_mask) == 1)[0]

            data_list = [data_list[i] for i in selected_particles]

            # also check if the edges are not mapped to 1?
            # This would only work for fcc, basically check if we only have 1's as edge_vec length
            # index_of_perfect_fcc = np.array(
            #     [
            #         i
            #         for i, g in enumerate(data_list)
            #         if len(g.edge_vec.norm(dim=1).unique()) == 1 and g.edge_vec.norm(dim=1).mean() == 1
            #     ]
            # )

            # data_list = [data_list[i] for i in index_of_perfect_fcc]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SyntheticDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        skeleton_graph_path,
        atom_types_path,
        pre_modification=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        one_hot_dim=5,
    ):
        self.skeleton_graph_path = skeleton_graph_path
        self.atom_types_path = atom_types_path
        self.pre_modification = pre_modification
        self.one_hot_dim = one_hot_dim
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

    def get_skeleton_graph(self):
        graph = torch.load(self.skeleton_graph_path)

        if self.pre_modification is not None:
            for modif in self.pre_modification:
                graph = modif(graph)

        return graph

    def get_atomic_types(
        self,
    ):
        all_types = np.array(torch.load(self.atom_types_path))

        # to use same network, we choose a 5 lengh encoder so need to add the zeros
        if all_types.shape[-1] != self.one_hot_dim:
            all_types = np.pad(
                all_types,
                ((0, 0), (0, 0), (0, self.one_hot_dim - all_types.shape[-1])),
                mode="constant",
            )
        return all_types

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        graph = self.get_skeleton_graph()
        all_types = self.get_atomic_types()

        for idx, atom_types in tqdm(enumerate(all_types), desc="Dataset"):
            graph = graph.clone()
            graph["index"] = idx

            # setting our skeleton graph atom types to be the idx permuation

            graph["node_attr"] = torch.tensor(atom_types, dtype=torch.float64)

            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
