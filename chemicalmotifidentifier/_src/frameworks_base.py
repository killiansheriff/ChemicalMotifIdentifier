import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from simplex import Simplex
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_chunked
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import Compose
from tqdm import tqdm

from .dataset import *
from .ml import ModelInference
from .models import NetworkForAGraphWithAttributesPeriodic
from .transforms import AddEdges, ModuloConcentration, RemoveCentralNode
from .wc import wc_from_conc_centers


# Base class
class BaseChemicalMotifIdentifier:
    """Equivariant Crystal Analyser (ECA) base class."""

    def __init__(self, crystal_structure):
        """Initiate the Equivariant Crystal Analyser (ECA) class.

        Args:
            crystal_structure (str): crystal structure of the system you want to analyse. Accepted variable are 'fcc', 'bcc' or 'hcp'.
        """
        assert crystal_structure in ["fcc", "bcc", "hcp"]

        self.crystal_structure = crystal_structure

        self.num_nodes_modes = {"fcc": 12, "hcp": 12, "bcc": 8}
        self.num_nodes = self.num_nodes_modes[self.crystal_structure]

        self.import_model_config()

        self.model = self.get_network()

    def import_model_config(self):
        raise ValueError('Needs to be implemented in children class.')
        # lmax, layers, outlength, number = 2, 2, 4, 0  # 2,2,4,0

        # model_config = {
        #     "out_feature_length": outlength,
        #     "max_radius": 3,  # 2.5
        #     "min_radius": 0,
        #     "number_of_basis": 10,
        #     "num_nodes": 12,  # only used for feqtur enormalization, we don't really care set it to a constant so that we can have only 1 network for all the crystal structure
        #     "num_neighbors": 5,  # only used for feqtur enormalization, we don't really care set it to a constant so that we can have only 1 network for all the crystal structure
        #     "layers": layers,
        #     "lmax": lmax,
        #     "net_number": number,
        #     "irreps_node_attr": "5x0e",
        #     "model_load": f"/home/ksheriff/PAPERS/second_paper/02_1nn_synthetic/data/nets/net_{lmax}-{layers}-{outlength}_{number}.pt",
        #     "mul": 3,  # 50
        # }
        # self.model_config = model_config

    def set_root(self, root):
        self.root = Path(root)

    def set_dataset_modes(self, dataset_config, dataset_object):
        self.dataset_config, self.dataset_object = dataset_config, dataset_object

    def set_model_config(self, model_config):
        self.model_config = model_config

    def set_dataset_config(self, dataset_config):
        self.dataset_config = dataset_config

    def set_concentrations(self, concs):
        self.concentrations = concs

    def set_atomic_types(self, atomic_types):
        self.atomic_types = atomic_types

    def get_dataset(self):
        if type(self.dataset_config.get("dump_file")) == list:
            dump_files = self.dataset_config["dump_file"]
            root = self.dataset_config["root"]
            ds = []
            for i, dump_file in enumerate(dump_files):
                self.dataset_config["dump_file"] = dump_file
                self.dataset_config["root"] = root + f"/runs/run_{i+1}/"
                ds.append(self.dataset_object(**self.dataset_config))

            ds = torch.utils.data.ConcatDataset(ds)
            self.dataset = ds

        else:  # string or none
            self.dataset = self.dataset_object(**self.dataset_config)

        self.batchsize = 4**4

    def get_concentrations(self):
        self.concentrations = torch.stack(
            [g.node_attr.sum(dim=0) for g in self.dataset], dim=0
        ).numpy()

    def get_central_atoms(self):
        self.central_atoms = torch.tensor(
            [g.central_atom for g in self.dataset]
        ).numpy()
        self.ntypes = np.max(self.central_atoms) + 1

    def get_atomic_types(self):
        self.atomic_types = torch.stack(
            [torch.argmax(g.node_attr, axis=1) for g in self.dataset], dim=0
        ).numpy()

    def get_concentration_before_permutation(self):
        self.concentration_before_permutation = np.array(
            [g.conc.numpy() for g in self.dataset]
        )

    def get_model(self):
        """Get a randomly intialized e3nn GNN.

        Returns:
            torch.nn.Model: randomly intialized e3nn model
        """

        net = NetworkForAGraphWithAttributesPeriodic(
            irreps_node_input="1x0e",  # Single scalar all set to one
            # One hot scalars (L=0 and even parity) on each atom to represent atom type
            irreps_node_attr=self.model_config["irreps_node_attr"],
            # Attributes in extra of the spherical harmonics: single scalar all set to one
            irreps_edge_attr="1x0e",
            # Single scalar (L=0 and even parity) to output (19 output with 1 speudo scaalr for parity)
            irreps_node_output=str(self.model_config["out_feature_length"])
            + "x0e",  # str(params['out_feature_length']-1)+"x0e+0o", #!!! if want speudo scalar
            max_radius=self.model_config[
                "max_radius"
            ],  # cutoff radius for convolution,
            min_radius=self.model_config["min_radius"],
            num_neighbors=self.model_config["num_neighbors"],
            num_nodes=self.model_config["num_nodes"],
            mul=self.model_config["mul"],  # multiplicity of irreducible representations
            layers=self.model_config[
                "layers"
            ],  # default 3 number of nonlinearities (number of convolutions = layers + 1)
            lmax=self.model_config[
                "lmax"
            ],  # 2 default maximum order of spherical harmonics
            pool_nodes=True,
            number_of_basis=self.model_config["number_of_basis"],
        )

        return net

    def get_network(self):
        """Initialize and load model weights. If model weights do not exists, generates and saves them.

        Returns:
            torch.nn.model: e3nn network model with loaded weights.
        """
        net = self.get_model()

        if not os.path.exists(self.model_config["model_load"]):
            print(
                f"Model {self.model_config['model_load']} does not exists. Creating it and saving it"
            )

            torch.save(net.state_dict(), self.model_config["model_load"])
            return net.double()

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = device
        net.load_state_dict(
            torch.load(self.model_config["model_load"], map_location=device)
        )
        net.double().to(device)
        return net

    def inference(self, **kwargs):
        """Performs inference of the graph dataset.

        Returns:
            np.array: output of the network forward path on each element of the dataset.
        """
        self.set_root(kwargs["root"])

        # self.root.rmdir()
        self.root.mkdir(exist_ok=True, parents=True)

        for key, value in kwargs.items():
            self.dataset_config[key] = value

        self.get_dataset()
        self.get_concentrations()
        self.get_atomic_types()

        y = ModelInference(
            model=self.model, dataset=self.dataset, batch_size=self.batchsize
        ).predict()
        return y

    def unique(self, y, rounding_number):
        """Round the embedings and unique

        Args:
            y (np.array): output of the network forward path on each element of the dataset.
            rounding_number (int): rounding number

        Returns:
            tuple (np.array, np.array): (rounded unique fingerprints, associated concentrations)
        """
        yhat, index, inverse, counts = np.unique(
            np.round(y, rounding_number),
            axis=0,
            return_index=True,
            return_counts=True,
            return_inverse=True,
        )
        concs = self.concentrations[index]
        atomic_types = self.atomic_types[index]
        return yhat, concs, counts, atomic_types, index, inverse


# Motif extraction for the synthetic dataset (used in "Chemical-motif characterization of SRO with E(3)-equivariant GNN")
class BaseSyntheticChemicalMotifIdentifier(BaseChemicalMotifIdentifier):
    """ECA class for running inference on the synthetic dataset.

    Args:
        ECA_base (cls): ECA_base class - requieres the crystal_structure argument.
    ```python
        structure = 'fcc'
        eca = ECA_Synthetic(crystal_structure=structure)
        df = eca.predict(
            root=root,
            skeleton_graph_path=skeleton_graph_path,
            atom_types_paths=atom_types_paths,
            nelement=nelement,
        )
        df.to_pickle(f"data/output/df_{structure}.pkl")
        ```
    """

    def __init__(self, **kwargs):
        """Initiate parameters of the class.

        Typical usage example:

        ```python
        structure = 'fcc'
        eca = ECA_Synthetic(crystal_structure=structure)
        df = eca.predict(
            root=root,
            skeleton_graph_path=skeleton_graph_path,
            atom_types_paths=atom_types_paths,
            nelement=nelement,
        )
        df.to_pickle(f"data/output/df_{structure}.pkl")
        ```
        """

        self.mode = "synthetic"
        super().__init__(**kwargs)

        dataset_config = {
            "transform": None,
            "pre_transform": None,
            "pre_filter": None,
            "one_hot_dim": 5,
        }
        self.pre_modifications = {
            "fcc": [RemoveCentralNode()],
            "bcc": [AddEdges(8), RemoveCentralNode()],
            "hcp": [RemoveCentralNode()],
        }

        dataset_object = SyntheticDataset
        self.set_dataset_modes(dataset_config, dataset_object)

    def scale(self, ys):
        # memory efficient way for finding max distance
        max_distance = 0
        for chunk_distances in pairwise_distances_chunked(
            ys.reshape(-1, 1), metric="euclidean", n_jobs=-1
        ):
            chunk_max_distance = np.max(chunk_distances)
            if chunk_max_distance > max_distance:
                max_distance = chunk_max_distance

        min_distance = 0
        min_max = max_distance - min_distance

        print(f"Max - min distance is {min_max}.")
        scaling = 1
        if max_distance != 0:
            scaling = 1.0 / min_max

        ys_scaled = ys * scaling
        return ys_scaled

    def get_structural_information_PCA(self, yhat, inverses, rounding_number=8):
        scaler = StandardScaler()

        scaler_yhat = scaler.fit_transform(yhat)
        pca_yhat = PCA(n_components=yhat.shape[1]).fit_transform(scaler_yhat)

        n_dim_struc = 1
        ys = np.zeros((len(yhat)))
        # count_not_good = 0

        pca_ndim = PCA(n_components=n_dim_struc)
        for inv in tqdm(inverses, desc="Inverses during generator space"):
            in_wc = np.where(inverses == inv)[0]

            ms_in_wc = pca_yhat[in_wc]

            if len(ms_in_wc) > 1:
                scaler_ms_in_wc = scaler.fit_transform(ms_in_wc)
                ys_wc = pca_ndim.fit_transform(scaler_ms_in_wc).reshape(1, -1)
            else:
                ys_wc = np.array([[0]])

            # if not len(np.unique(np.round(ys_wc, 8), axis=0)[0]) == len(ms_in_wc):

            # count_not_good += 1

            ys[in_wc] = ys_wc

        # print(f"Number of times were not good {count_not_good}")
        ys_scaled = self.scale(ys)
        ys_scaled = np.round(ys_scaled, rounding_number)

        return ys_scaled

    def get_structural_information_AE(self, yhat, rounding_number=8):
        from eca.dataset import TensorDataset
        from eca.models import Autoencoder
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        dataset = TensorDataset(yhat)
        dataloader = DataLoader(dataset, batch_size=640, shuffle=True)
        testloader = DataLoader(dataset, batch_size=640, shuffle=False)

        model = (
            Autoencoder(
                input_dim=yhat.shape[1],
                latent_dim=1,
                encoder_layers=[7, 4],
                decoder_layers=[7, 4],
            )
            .double()
            .to("cuda")
        )

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        checkpoint_path = None

        # train the autoencoder
        for epoch in tqdm(range(1000), desc="Epochs..."):
            for data in tqdm(dataloader, mininterval=30):
                data = data.to("cuda")
                ytrue = data
                optimizer.zero_grad()
                ypred = model(ytrue)
                loss = loss_fn(ypred, ytrue)
                loss.backward()
                optimizer.step()
            print("Epoch [{}/{}], Loss: {:.10f}".format(epoch + 1, 10, loss.item()))

        results = []
        with torch.no_grad():
            for batch in tqdm(testloader, desc="Inference"):
                inputs = batch.to("cuda")
                outputs = model.encoder(inputs)
                results.append(outputs.cpu().numpy())

        ys = np.concatenate(results, axis=0)

        ys = self.scale(ys)
        ys_rounded = np.round(ys, rounding_number).flatten()

        print(len(np.unique(yhat, axis=0)))
        print(len(np.unique(ys_rounded, axis=0)))

        return ys_rounded

    def get_structural_information(self, yhat, rounding_number=8):
        ys = self.scale(yhat)
        ys_rounded = np.round(ys, rounding_number)

        # print(len(np.unique(yhat, axis=0)))
        # print(len(np.unique(ys_rounded, axis=0)))
        return ys_rounded

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

        structural_information = self.get_structural_information(yhat)

        generator_space = np.hstack((cartesian_coords, structural_information))

        return generators, generator_space, inverses

    def apply_permutation(self, arr, perm):
        indices = np.argsort(perm)
        return indices[arr]

    def generate_space(
        self, generators, generator_space, concs, inverses, yhat, counts, atomic_types
    ):
        """Extend the generator space to the full concentration space.

        Args:
            generators (np.array): concentration generators
            generator_space (np.array): concentration generators space
            concs (np.array): concentrations associated with each yhat.
            inverses (np.array): inverses for each microstates / in wich concentration class they belong wrt to concentration.
            yhat (np.array): ounded fingerprint of microstate having generating concentrations.
            counts (np.array): multiplicity of microstates that allows generators concentrations.

        Returns:
            dict: key have the form (x1,....,x5,x6,..,x10) and map to ((x*1,....,x*4,x*5)) where x1,....,x5 are concentration (barycentric coordinates) and x6,..,x10 the output of the NN on the chemical shell modulo the generators, and x*1,....,x*4 are the cartesian coordinate encoding concentration information and x*5 encodes structural information.
        """
        concs = concs[:, : self.nelement]

        # Generate all permutations of the generators
        n_generators = generators.shape[0]

        generator_perms = [list(itertools.permutations(gen)) for gen in generators]
        permutations = np.array(
            [
                list(itertools.permutations(range(self.nelement)))
                for _ in range(len(generator_perms))
            ]
        )

        generator_data = [
            np.unique(g, axis=0, return_index=True) for g in generator_perms
        ]
        generator_perms = [data[0] for data in generator_data]

        generator_indices = [
            perm[data[1]] for perm, data in zip(permutations, generator_data)
        ]

        # Convert generator permutations to cartesian coordinates
        gen_coords = [
            self.simplex.barycenter_coods_2_cartesian(self.vertices, coords)
            for coords in generator_perms
        ]

        extended_space = []
        concentrations = []
        countss = []
        yhats = []
        atomic_typess = []
        for idx in range(n_generators):
            # Select all microstate sharing same concentration in the generator space

            in_generator = inverses == idx
            ms_in_column = generator_space[in_generator]
            y = yhat[in_generator]

            count = counts[in_generator]
            atomic_type = atomic_types[in_generator]

            # loop over it's generated permutations
            for idy, gen_perm in enumerate(generator_perms[idx]):
                generated_ms = np.copy(ms_in_column)

                # set good concentration cooridnates for all ms in the class
                generated_ms[:, : generators.shape[1] - 1] = np.full(
                    (len(ms_in_column), self.nelement - 1), gen_coords[idx][idy]
                )

                concentrations.append(
                    np.full((len(ms_in_column), *gen_perm.shape), gen_perm)
                )
                yhats.append(y)
                countss.append(count)

                # need to apply the correction to permutatx.isdisjoint(y)

                unmodulo_atomic_type = self.apply_permutation(
                    np.copy(atomic_type), generator_indices[idx][idy]
                )

                atomic_typess.append(unmodulo_atomic_type)

                extended_space.append(generated_ms)

        extended_space = np.concatenate(extended_space, axis=0)
        concentrations = np.concatenate(concentrations, axis=0)
        self.set_concentrations(concentrations)
        yhats = np.concatenate(yhats, axis=0)
        countss = np.concatenate(countss, axis=0)
        atomic_typess = np.concatenate(atomic_typess, axis=0)

        # this is the concentration of our chemical shell (not modulo) and the fingerprint of the graph (modulo the concentration). Given this valye we can map to phys constrained embeding space

        emb = np.hstack((concentrations, yhats))
        emb = [tuple(i) for i in emb]

        extended_space = [tuple(i) for i in extended_space]

        # DATAFRAME
        df = pd.DataFrame(
            data={
                "shell_ID": np.arange(len(extended_space)),
                "shell_phys_emb": extended_space,
                "counts": countss,
                "shell_atomic_types": [tuple(a) for a in atomic_typess],
                "shell_concentration": [tuple(c) for c in concentrations],
            },
            index=pd.MultiIndex.from_tuples(emb),
        )

        nms = len(np.unique(extended_space, axis=0))
        print(f"Unique en phys emb space gives out {nms} microstates.")
        # df.index.names = ["c1", "c2", "c3", "c4", "c5", "y1", "y2", "y3", "y4"]

        return df

    def predict(self, root, skeleton_graph_path, atom_types_paths, nelement):
        """Routine to get self.decoder which is a dict in which key have the form (x1,....,x5,x6,..,x10) and map to ((x*1,....,x*4,x*5)) where x1,....,x5 are concentration (barycentric coordinates) and x6,..,x10 the output of the NN on the chemical shell modulo the generators, and x*1,....,x*4 are the cartesian coordinate encoding concentration information and x*5 encodes structural information.

        Args:
            root (str): root directory to save dataset objects.
            skeleton_graph_path (str): skeleton graph path
            atom_types_paths (list): list of path files that contains all permutations giving concentrations generators.
            nelement (int): number of element allowed in synthetic dataset.
        """
        y_list = []
        concs_list = []
        atomic_types_list = []
        counts_list = []

        for idx, atom_types_path in tqdm(
            enumerate(atom_types_paths), desc="Sub atom types"
        ):
            y = self.inference(
                root=root + f"{idx}/",
                atom_types_path=atom_types_path,
                skeleton_graph_path=skeleton_graph_path,
                pre_modification=self.pre_modifications[self.crystal_structure],
            )

            yhat, concs, counts, atomic_types, _, _ = self.unique(y, rounding_number=8)

            concs_list.append(concs)
            atomic_types_list.append(atomic_types)
            y_list.append(yhat)
            counts_list.append(counts)

        del self.dataset
        del self.model
        print("loop done")

        self.set_concentrations(np.concatenate(concs_list, axis=0))
        del concs_list

        self.set_atomic_types(np.concatenate(atomic_types_list, axis=0))
        del atomic_types_list

        y = np.concatenate(y_list, axis=0)
        del y_list

        print("concat done")

        yhat, concs, _, atomic_types, _, inverses = self.unique(y, rounding_number=8)
        del y
        print("last unique done")
        torch.save(yhat, root + "yhat.pt")

        counts_with_duplicates = np.concatenate(counts_list, axis=0)
        counts = np.array(
            [np.sum(counts_with_duplicates[inverses == i]) for i in range(len(yhat))]
        )

        del counts_with_duplicates

        # print(f" We have {len(yhat)} microstates in the generator dataset.")

        generators, generator_space, inverses = self.set_up_generator_space(
            yhat,
            concs,
            nelement=nelement,
        )
        print("Generator space done")
        df = self.generate_space(
            generators, generator_space, concs, inverses, yhat, counts, atomic_types
        )
        # print(f" We have {len(df)} microstates after extension.")
        print("df done")

        return df


# Motif extraction for the atomistic data (used in "Chemical-motif characterization of SRO with E(3)-equivariant GNN")
class BaseMonteCarloChemicalMotifIdentifier(BaseChemicalMotifIdentifier):
    """ECA_MD class to operates on MD/MC simulations outputs (e.g dump files).

    Args:
        ECA_base (cls): ECA_base class - requieres the crystal_structure argument.
    ```python
        structure='fcc'
        dump_files = glob.glob('*.dump')

        eca = ECA_MD(crystal_structure=structure)
        for i, dump_file in enumerate(dump_files):
            df = eca.predict(root=root, dump_file=dump_file)
            kl = eca.get_kl(df)
    ```
    """

    def __init__(self, **kwargs):
        """Initiate parameters of the class.

        Typical usage example:

        ```python
        structure='fcc'
        dump_files = glob.glob('*.dump')

        eca = ECA_MD(crystal_structure=structure)
        for i, dump_file in enumerate(dump_files):
            df = eca.predict(root=root, dump_file=dump_file)
            kl = eca.get_kl(df)
        ```
        """
        self.mode = "synthetic"
        super().__init__(**kwargs)

        dataset_config = {"pre_filter": None, "one_hot_dim": 5}

        dataset_object = OvitoData
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

    def import_synthetic(self):
        """Import chemical shell synthetic dataset pandas dataframe"""
        raise ValueError('Needs to be implemented in children class.')
        # self.df_synthetic = pd.read_pickle(
        #     f"/home/ksheriff/PAPERS/second_paper/02_1nn_synthetic/data/output/df_{self.crystal_structure}.pkl"
        # )

    def set_information_theory(self, df):
        """Set information theory probabilities to compute entropy, KL, and CE.

        Args:
            df (dataframe): pandas dataframe with the detected microstates.
        """
        p = df["count_md"] / df["count_md"].sum()
        q = df["counts"] / self.ntypes ** (
            self.num_nodes + 1
        )  #!!! prob of seeing a local chem motif with central atom so norm that they don't sum to 1
        df["p"] = p
        df["q"] = q
        df["pop_ratio"] = df["p"] / df["q"]

    def get_per_atom_stats(self, arr, index, inverses):
        stats = []
        for idx in index:
            inv = inverses[idx]
            mask = inverses == inv
            data = arr[mask]
            stat = (np.mean(data), np.std(data, ddof=1))
            stats.append(stat)
        stats = np.array(stats)
        return stats[:, 0], stats[:, 1]

    def set_df_unique(self, df_unique):
        self.df_unique = df_unique

    def predict(
        self,
        root,
        dump_file,
        frame_number=0,
        rounding_edge_vec_value=True,
        rounding_number=8,
        **kwargs,
    ):
        """Predict microstate distributions in dump_file.

        Args:
            root (str): root folder for dataset
            dump_file (str): dump file path
            frame_number (int, optional): dump file frame number. Defaults to 0.
            rounding_edge_vec_value (bool, optional): Will set edge value to exact numbers. Defaults to True.
            rounding_number (int, optional): Rounding number for network fingerprints. Defaults to 8.

        Returns:
            pd.DataFrame: numpy dataframe with identified microstates.
        """

        y = self.inference(
            root=root,
            dump_file=dump_file,
            nneigh=self.num_nodes,
            frame_number=frame_number,
            rounding_edge_vec_value=rounding_edge_vec_value,
            pre_transform=self.transfroms[self.crystal_structure],
        )
        np.save(self.root / "preds.npy", y)

        self.get_central_atoms()
        self.get_concentration_before_permutation()

        y = np.hstack(
            (
                self.central_atoms.reshape(-1, 1),
                self.concentration_before_permutation,
                np.round(y, rounding_number),
            )
        )

        df = self.df_synthetic.loc[
            pd.MultiIndex.from_tuples([tuple(yi)[1:] for yi in y])
        ]

        shell_phys_emb = np.array(list(df["shell_phys_emb"].values))

        phys_emb = np.hstack((self.central_atoms.reshape(-1, 1), shell_phys_emb))

        np.save(self.root / "phys_emb.npy", phys_emb)

        phys_emb_unique, index, inverses, counts = np.unique(
            phys_emb, axis=0, return_counts=True, return_index=True, return_inverse=True
        )

        df_unique = df.iloc[index].copy()

        if kwargs.get("add_displacement") == True:
            disp = np.array([g.disp.numpy() for g in self.dataset]).flatten()
            df["d"] = disp
            d_mean, d_std = self.get_per_atom_stats(disp, index, inverses)
            df_unique["d_mean"], df_unique["d_std"] = d_mean, d_std

        if kwargs.get("add_energies") == True:
            energies = np.array([g.energy.numpy() for g in self.dataset]).flatten()
            df["e"] = energies
            e_mean, e_std = self.get_per_atom_stats(energies, index, inverses)
            df_unique["e_mean"], df_unique["e_std"] = e_mean, e_std

        df_unique["central_atom_type"] = self.central_atoms[index]
        df_unique["count_md"] = counts
        df_unique["emb"] = df_unique.index

        df_unique = df_unique.set_index(
            pd.MultiIndex.from_tuples(
                [
                    (center, idx)
                    for center, idx in zip(
                        df_unique["central_atom_type"], df_unique["shell_ID"]
                    )
                ]
            )
        )
        df_unique.index.names = ["central_atom_type", "shell_ID"]

        # saving full df with all atoms before detecting microstates.
        df.to_pickle(self.root / "df.pkl")
        df_unique.to_pickle(self.root / "df_microstates.pkl")

        return df_unique

    def get_kl(self, df):
        self.set_information_theory(df)
        kl = df["p"] * np.log2(df["p"] / df["q"])

        kl = kl.sum()

        np.savetxt(
            self.root / "kl.txt",
            [kl],
            fmt="%7.5f",
            header="KL (bits)",
        )
        return kl

    def get_wc(
        self,
    ):
        indices_columns_0 = np.all(self.concentration_before_permutation == 0, axis=0)
        concs = self.concentration_before_permutation[:, ~indices_columns_0]

        n_principal = len(concs[0])

        wc_parameters = wc_from_conc_centers(
            concentrations=concs,
            centers=self.central_atoms + 1,
            ncomponant=n_principal,
            natoms_shell=self.num_nodes,
        )

        np.save(
            self.root / f"wc_{n_principal}x{n_principal}.npy",
            wc_parameters,
        )
        return wc_parameters
