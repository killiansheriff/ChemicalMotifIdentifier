import itertools

import numpy as np


class Simplex:
    """Implement the geometric object of a simplex (https://en.wikipedia.org/wiki/Simplex)

    Raises:
        ValueError: only implemented n-simplex for n >=2 and n<=4.

    Example of usage:
       ```python
        triangle = Simplex(n_dim=2, edge_length=1, nneigh=12)
        concentrations, coords = triangle.get_mapping()
        generators, counts = triangle.get_generators(concentrations)
       ```
    """

    def __init__(self, n_dim, edge_length=1, nneigh=12):
        """_summary_

         Args:
             n_dim (int): dimension of the simplex to create. Usually nelement - 1.
             edge_length (int, optional): edge length of the simplex. Defaults to 1.
             nneigh (int, optional): number of neighbors in the chemical shell space (governs how many concentration class we have in the chemical shell WC space). Defaults to 12.

         Example of usage:
        ```python
         triangle = Simplex(n_dim=2, edge_length=1, nneigh=12)
         concentrations, coords = triangle.get_mapping()
         generators, counts = triangle.get_generators(concentrations)
        ```
        """
        self.n_dim = n_dim
        self.edge_length = edge_length
        self.nneigh = nneigh
        self.vertex_functions = {
            1: self.get_line_segment_vertices,
            2: self.get_equilateral_triangle_vertices,
            3: self.get_regular_tetrahedron_vertices,
            4: self.get_simplex_4_vertices,
        }

    def get_line_segment_vertices(self):
        """Return vertices of a line segement.

        Returns:
            np.array: vertices.
        """
        vertices = np.array(
            [
                [0],
                [self.edge_length],
            ]
        )
        return vertices

    def get_regular_tetrahedron_vertices(self):
        """Return vertices of a regular tetrahedron (https://mathworld.wolfram.com/RegularTetrahedron.html).

        Returns:
            np.array: vertices.
        """
        #
        vertices = np.array(
            [
                [1 / 3 * np.sqrt(3) * self.edge_length, 0, 0],
                [-1 / 6 * np.sqrt(3) * self.edge_length, +1 / 2 * self.edge_length, 0],
                [-1 / 6 * np.sqrt(3) * self.edge_length, -1 / 2 * self.edge_length, 0],
                [0, 0, 1 / 3 * np.sqrt(6) * self.edge_length],
            ]
        )
        return vertices

    def get_equilateral_triangle_vertices(self):
        """Return vertices of a triangle.

        Returns:
            np.array: vertices.
        """
        sqrt_3 = np.sqrt(3)
        vertices = np.array(
            [
                [0, 0],
                [self.edge_length, 0],
                [self.edge_length / 2, sqrt_3 / 2 * self.edge_length],
            ]
        )

        return vertices

    def get_simplex_4_vertices(self):
        """Return vertices of a 4-simplex (https://polytope.miraheze.org/wiki/Pentachoron).

        Returns:
            np.array: vertices.
        """

        vertices = np.array(
            [
                [0, 0, 0, np.sqrt(10) / 5],
                [0, 0, np.sqrt(6) / 4, -np.sqrt(10) / 20],
                [0, np.sqrt(3) / 3, -np.sqrt(6) / 12, -np.sqrt(10) / 20],
                [1 / 2, -np.sqrt(3) / 6, -np.sqrt(6) / 12, -np.sqrt(10) / 20],
                [-1 / 2, -np.sqrt(3) / 6, -np.sqrt(6) / 12, -np.sqrt(10) / 20],
            ]
        )
        return vertices

    def get_vertex_coordinates(self):
        """Get vertex coordinates

        Raises:
            ValueError: The {self.n_dim}-simplex has not been implemented.

        Returns:
            np.array: cooridnates of vertices.
        """
        if self.n_dim not in self.vertex_functions:
            raise ValueError(f"The {self.n_dim}-simplex has not been implemented.")
        return self.vertex_functions[self.n_dim]()

    def get_barycentric_coords(self):
        """Get barycentric coordinates given the constraint that they must sum to nneigh.

        Returns:
            np.array: barycentric coordinates.
        """
        constraint = self.nneigh
        coords = np.array(
            list(itertools.product(range(self.nneigh + 1), repeat=self.n_dim + 1))
        )
        if constraint is not None:
            coords = coords[np.sum(coords, axis=1) == constraint]
        return coords

    def barycenter_coods_2_cartesian(self, vertices, coords):
        """Convert barycentric coordinates to cartesian coordinates

        Args:
            vertices (np.array): cartesian coordinates of vertices
            coords (np.array): barycentric coordinates we want to express in cartesian

        Returns:
            np.array: cartesiam representation of the barycentric coords.
        """
        return np.dot(coords, vertices)

    def get_mapping(self):
        """Get mapping from barycenter coordinates to cartesian for all possible concentrations.

        Returns:
            tuple: (bary_coords,cartesian_coords)
        """
        vertices = self.get_vertex_coordinates()
        bary_coords = self.get_barycentric_coords()
        cartesian_coords = self.barycenter_coods_2_cartesian(vertices, bary_coords)

        assert np.isclose(
            np.linalg.norm(cartesian_coords[0] - cartesian_coords[-1]),
            self.nneigh,
            atol=0.1,
        )

        return bary_coords, cartesian_coords

    def get_generators(self, bary_coords):
        """Return the barycenter coordinates that can generated the whole space by just swaping elements.

        Args:
            bary_coords (np.array): barycenter coordinates of the whole space

        Returns:
            tuple: (barycenter coordinates of the generators, number of element they generate)
        """
        generators, counts = np.unique(np.sort(bary_coords), axis=0, return_counts=True)
        return generators, counts

    def barycentric_2_cartesian(self, bary_coords):
        vertices = self.get_vertex_coordinates()
        cartesian_coords = self.barycenter_coods_2_cartesian(vertices, bary_coords)
        return cartesian_coords
