import numpy as np
from ovito.io import import_file
from tqdm import tqdm

from NshellFinder import NshellFinder


def concat_nshell_neighbor_idx(dump_files, cutoff, crystal_structure, nframe=0):
    # Get and shift the neighbor indices based on how many dump file I have
    Ndump = len(dump_files)
    indices = []

    count = []

    for i in tqdm(range(Ndump), desc="Dump i"):
        pipe = import_file(
            dump_files[i]
        )  # don't import all at same type -- issue with multiple frame
        pipe.modifiers.append(
            NshellFinder(cutoff=cutoff, crystal_structure=crystal_structure)
        )

        data = pipe.compute(nframe)

        count.append(data.particles.count)
        nn = data.attributes["Neighbor indices per shell"]

        indices.append(nn)

    max_nshell = len(indices[0])

    indices_n = []
    for n in range(max_nshell):
        indices_i = []
        for i in range(Ndump):
            shifted_idx = indices[i][n] + i * count[i]
            indices_i.append(shifted_idx)
        indices_n.append(np.concatenate(indices_i))

    return indices_n


# if __name__ == "__main__":
#     dump_files = [
#         f"/home/ksheriff/PAPERS/first_paper/03_mtp/data/dumps/dumps_mtp_mc/ordered_relaxation_20_{i}_300K.dump"
#         for i in range(1, 216 + 1)
#     ]
#     shifted_nshell_indices = concat_nshell_neighbor_idx(
#         dump_files=dump_files, cutoff=18.2, crystal_structure="fcc"
#     )
