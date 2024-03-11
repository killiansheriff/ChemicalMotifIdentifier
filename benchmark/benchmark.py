import time

import matplotlib.pyplot as plt
from eca import ECA_MD_Thermal

plt.style.use("paper")


import os

import GenerateRandomSolution as grs
import numpy as np
from ase.build import bulk
from ovito.io import export_file
from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource


def generate_random_system(crystal_structure: str, size: tuple):
    os.makedirs("data/dumps/", exist_ok=True)

    # Create a full Cu system
    atoms = bulk(name="Cu", crystalstructure=crystal_structure, a=1, cubic=True).repeat(
        size
    )

    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))

    # Apply the RSS modifier based on the above wanted concentration
    pipeline.modifiers.append(
        grs.GenerateRandomSolution(
            only_selected=False,
            concentrations=[1 / 3, 1 / 3, 1 / 3],
            seed=np.random.randint(1000000),
        )
    )

    data = pipeline.compute()

    export_file(
        data,
        f"data/dumps/{crystal_structure}_{size}.dump",
        "lammps/dump",
        columns=[
            "Particle Identifier",
            "Particle Type",
            "Position.X",
            "Position.Y",
            "Position.Z",
        ],
    )


if __name__ == "__main__":
    structure = "fcc"
    nrange = np.arange(5, 60, 5)

    for n in nrange:
        generate_random_system(structure, size=(n, n, n))

    dump_files = [f"data/dumps/fcc_{(n,n,n)}.dump" for n in nrange]
    times = []

    eca = ECA_MD_Thermal(crystal_structure=structure, rmsd_cutoff=0.05)
    for i, dump_file in enumerate(dump_files):
        t = time.perf_counter()
        root = f"data/eca_id/dump_{i}/"
        df = eca.predict(root=root, dump_file=dump_file)
        kl = eca.get_kl(df)
        df.to_pickle(root + "df_microstates.pkl")
        times.append(time.perf_counter() - t)

    fig, ax = plt.subplots()
    ax.plot(nrange**3 * 4, times, "-o")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Number of atoms")
    fig.savefig("number_of_atoms_vs_time.pdf")

    # Deleting
    os.system(f"rm -rf data/eca_id")