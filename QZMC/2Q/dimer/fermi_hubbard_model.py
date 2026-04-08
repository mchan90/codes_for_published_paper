
# modified by mchan to different ordering, which allows two-qubit reduction with
# a qiskit's ParityMapper (at qiskit version 0.6)

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""The Fermi-Hubbard model"""
import numpy as np

from qiskit_nature.second_q.operators import FermionicOp

from qiskit_nature.second_q.hamiltonians.lattice_model import LatticeModel
from qiskit_nature.second_q.hamiltonians.lattices import Lattice


class FermiHubbardModel(LatticeModel):
    r"""The Fermi-Hubbard model.

    This class implements the following Hamiltonian:

    .. math::
        H = \sum_{i, j}\sum_{\sigma = \uparrow, \downarrow} t_{i, j} c_{i, \sigma}^\dagger
        c_{j, \sigma} + U \sum_{i} n_{i, \uparrow} n_{i, \downarrow},

    where :math:`c_{i, \sigma}^\dagger` and :math:`c_{i, \sigma}` are creation and annihilation
    operators of a fermion at the site :math:`i` with spin :math:`\sigma`. The operator :math:`n_{i,
    \sigma}` is the number operator, which is defined by :math:`n_{i, \sigma} = c_{i,
    \sigma}^\dagger c_{i, \sigma}`. The matrix :math:`t_{i, j}` is a Hermitian matrix called the
    interaction matrix. The parameter :math:`U` represents the strength of the on-site interaction.

    This model is instantiated using a
    :class:`~qiskit_nature.second_q.hamiltonians.lattices.Lattice`. For example, using a
    :class:`~qiskit_nature.second_q.hamiltonians.lattices.LineLattice`:

    .. code-block:: python

        line_lattice = LineLattice(num_nodes=10, boundary_condition=BoundaryCondition.OPEN)

        fermi_hubbard_model = FermiHubbardModel(
            line_lattice.uniform_parameters(
                uniform_interaction=-1.0,
                uniform_onsite_potential=0.0,
            ),
            onsite_interaction=5.0,
        )
    """

    r""" ordering convention is changed to spin up 1, 2, 3 ... then spin down 1, 2, 3,..
    This is because orginal convention makes unnecessary complication of qubit operators.
    """

    def __init__(self, lattice: Lattice, onsite_interaction: complex) -> None:
        """
        Args:
            lattice: Lattice on which the model is defined.
            onsite_interaction: The strength of the on-site interaction.
        """
        super().__init__(lattice)
        self._onsite_interaction = onsite_interaction

    def hopping_matrix(self) -> np.ndarray:
        """Return the hopping matrix."""
        return self.interaction_matrix()

    @property
    def register_length(self) -> int:
        return 2 * self._lattice.num_nodes

    def second_q_op(self) -> FermionicOp:
        """Return the Hamiltonian of the Fermi-Hubbard model in terms of ``FermionicOp``.

        Returns:
            FermionicOp: The Hamiltonian of the Fermi-Hubbard model.
        """
        kinetic_ham = {}
        interaction_ham = {}
        weighted_edge_list = self._lattice.weighted_edge_list
        register_length = 2 * self._lattice.num_nodes
        # kinetic terms
        for spin in range(2):
            for node_a, node_b, weight in weighted_edge_list:
                if node_a == node_b:
                    index = spin * self._lattice.num_nodes + node_a
                    kinetic_ham[f"+_{index} -_{index}"] = weight

                else:
                    if node_a < node_b:
                        index_left = spin * self._lattice.num_nodes + node_a
                        index_right = spin * self._lattice.num_nodes + node_b
                        hopping_parameter = weight
                    elif node_a > node_b:
                        index_left = spin * self._lattice.num_nodes + node_b
                        index_right = spin * self._lattice.num_nodes + node_a
                        hopping_parameter = np.conjugate(weight)
                    kinetic_ham[f"+_{index_left} -_{index_right}"] = hopping_parameter
                    kinetic_ham[f"-_{index_left} +_{index_right}"] = -np.conjugate(
                        hopping_parameter
                    )
        # on-site interaction terms
        for node in self._lattice.node_indexes:
            index_up = node
            index_down = self._lattice.num_nodes + node
            interaction_ham[
                f"+_{index_up} -_{index_up} +_{index_down} -_{index_down}"
            ] = self._onsite_interaction

        ham = {**kinetic_ham, **interaction_ham}

        return FermionicOp(ham, num_spin_orbitals=register_length, copy=False)
