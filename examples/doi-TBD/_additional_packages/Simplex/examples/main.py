import simplex as sp

triangle = sp.Simplex(n_dim=2, edge_length=1, nneigh=12)
concentrations, coords = triangle.get_mapping()
generators, counts = triangle.get_generators(concentrations)
breakpoint()
