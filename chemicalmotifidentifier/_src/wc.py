import numpy as np
from numpy import *


def wc_from_conc_centers(concentrations, centers, ncomponant, natoms_shell):
    """Works only for equiatomic or need to modify ci

    Args:
        concentrations (_type_): _description_
        centers (_type_): _description_
        ncomponant (_type_): _description_
        natoms_shell (_type_): _description_

    Returns:
        _type_: _description_
    """

    n = {}
    c = {}
    for i in range(1, ncomponant + 1, 1):
        n[i] = len(np.where(centers == i)[0])
        c[i] = n[i] / len(centers)

    wc = {}
    for i in range(1, ncomponant + 1):
        for j in range(1, ncomponant + 1):
            sub_conc = concentrations[np.where(centers == j)[0]]

            nij = sub_conc[:, i - 1].sum() / (natoms_shell)

            ci = c[
                i
            ]  # 1 /ncomponant  #!!! need to change to allow other concentrations

            wc[(i, j)] = 1 - 1 / ci * (nij / n[j])

    wc = np.array(list(wc.values())).reshape(ncomponant, ncomponant)
    return wc
