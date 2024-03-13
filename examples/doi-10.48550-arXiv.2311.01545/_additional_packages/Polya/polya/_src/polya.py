import numpy as np
import sympy as sp
from sympy.combinatorics.perm_groups import PermutationGroup

from .graphs import (
    Bcc1n2nnnGraph,
    Bcc1nnGraph,
    Fcc1nn2nn3nnGraph,
    Fcc1nn2nnGraph,
    Fcc1nnGraph,
    Hcp1nnGraph,
)

GRAPHS = {
    "fcc": Fcc1nnGraph,
    "bcc": Bcc1nnGraph,
    "fcc_1nn2nn": Fcc1nn2nnGraph,
    "hcp": Hcp1nnGraph,
    "fcc_1nn2nn3nn": Fcc1nn2nn3nnGraph,
    "bcc_1nn2nn": Bcc1n2nnnGraph,
}


class Polya:
    """Class to get Polya's pattern inventory.

    ```python
    from polya import Polya
    pl = Polya(graph_name="fcc")
    p_g, nms = pl.get_gt(ntypes=3)
    print(p_g)
    ```
    """

    def __init__(self, graph_name):
        """Class to get Polya's pattern inventory.

        ```python
        from polya import Polya
        pl = Polya(graph_name="fcc")
        p_g, nms = pl.get_gt(ntypes=3)
        print(p_g)
        ```
        """
        self.graph = GRAPHS[graph_name]()

    def get_cycle_index(self, permgroup):
        cycle_types = [p.cycle_structure for p in permgroup]
        monomials = [
            np.prod(
                [sp.symbols(f"s_{ctype}") ** cycle[ctype] for ctype in cycle.keys()]
            )
            for cycle in cycle_types
        ]
        nnodes = np.sum([key * value for key, value in cycle_types[0].items()])
        group_size = len(permgroup) + 1  # add identity
        cycle_index = np.sum(monomials) + sp.symbols("s_1") ** nnodes
        return cycle_index / group_size  # need divided size of group

    def get_gt(self, ntypes):
        """Get cycle index polynomial and number of distinct graphs.

        Ins:
            ntypes (int) = Number of atom types

        Returns:
            p_g = cycle index polynomial
            nms = number of distinct graphs while taking into account symmeteries
        """

        # Get graph object
        self.g = self.graph.get_graph()

        nnodes = self.g.vcount()

        # Compute the automorphism group
        permgroup = np.array(self.g.get_automorphisms_vf2())

        # Get the permutation representation of the group
        permgroup = PermutationGroup(permgroup)
        cycle_index = self.get_cycle_index(permgroup)

        # define symbolic variables for d1 to d10
        types = sp.symbols(f"t1:{ntypes+1}")

        # replace s_i with the sum of the powers of the d variables and factorize
        p_g = sp.factor(
            cycle_index.subs(
                [
                    (
                        sp.symbols(f"s_{i}"),
                        np.sum([types[j] ** i for j in range(ntypes)]),
                    )
                    for i in range(1, nnodes + 1)
                ]
            )
        )

        # replace s_i with the sum of the powers of 1 for each variable and factorize
        nms = sp.factor(
            cycle_index.subs(
                [
                    (sp.symbols(f"s_{i}"), sum([1**i for _ in range(ntypes)]))
                    for i in range(1, nnodes + 1)
                ]
            )
        )
        return p_g, nms
