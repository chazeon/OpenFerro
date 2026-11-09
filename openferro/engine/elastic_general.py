"""
General-symmetry elastic energy terms.

The symmetry of the crystal is set by the full stiffness tensor ``C_ijkl``
that is passed in, not by the code. The same two engines therefore serve
cubic, tetragonal, orthorhombic, or lower-symmetry materials. The cubic
engines in :mod:`openferro.engine.elastic` are left unchanged.

Conventions
-----------
* Strain is infinitesimal and engineering Voigt, ``(e_xx, e_yy, e_zz,
  2e_yz, 2e_xz, 2e_xy)``.
* Stiffness is given as a 6x6 Voigt matrix or the full ``(3, 3, 3, 3)``
  tensor, in eV/Angstrom^3. ``C_voigt[I, J]`` uses the standard Voigt map
  ``11->1, 22->2, 33->3, 23->4, 13->5, 12->6``, with no factor of 2 on the
  shear entries.
* The lattice is orthogonal with constants ``(a1, a2, a3)`` in Angstrom.

Notes
-----
This file is part of OpenFerro.
"""

import itertools

import jax.numpy as jnp
import numpy as np


# Voigt index of the unordered pair (i, j), with i, j in {0, 1, 2}.
_VOIGT_INDEX = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])

# Index pairs (i, j) for Voigt slots 0..5, used by tensor_to_voigt.
_VOIGT_PAIRS_I = np.array([0, 1, 2, 1, 0, 0])
_VOIGT_PAIRS_J = np.array([0, 1, 2, 2, 2, 1])

# 1 eV/Angstrom^3 = 1.602176634e11 Pa = 160.2176634 GPa (exact SI definition
# of the eV with CODATA 2018 elementary charge).
GPA_PER_EV_PER_ANGSTROM3 = 160.2176634


def _check_voigt_matrix(C_voigt):
    """Raise ValueError unless ``C_voigt`` is a symmetric (6, 6) array."""
    C_np = np.asarray(C_voigt)
    if C_np.shape != (6, 6):
        raise ValueError("Voigt stiffness must have shape (6, 6).")
    if not np.allclose(C_np, C_np.T):
        raise ValueError("Voigt stiffness must be symmetric.")


def voigt_to_tensor(C_voigt):
    """
    Convert a 6x6 Voigt stiffness matrix to the full ``(3, 3, 3, 3)`` tensor.

    Parameters
    ----------
    C_voigt : array_like
        Symmetric stiffness matrix, shape ``(6, 6)``, in the Voigt order above.

    Returns
    -------
    jnp.ndarray
        Stiffness tensor ``C[i, j, k, l]``, shape ``(3, 3, 3, 3)``. It has
        minor symmetry ``C[i,j,k,l] = C[j,i,k,l] = C[i,j,l,k]`` and major
        symmetry ``C[i,j,k,l] = C[k,l,i,j]`` by construction.

    Raises
    ------
    ValueError
        If the input is not a symmetric ``(6, 6)`` array.
    """
    _check_voigt_matrix(C_voigt)
    C_voigt = jnp.asarray(C_voigt)
    idx = _VOIGT_INDEX
    return C_voigt[idx[:, :, None, None], idx[None, None, :, :]]


def tensor_to_voigt(C_tensor):
    """
    Convert a full ``(3, 3, 3, 3)`` stiffness tensor to the 6x6 Voigt matrix.

    Parameters
    ----------
    C_tensor : array_like
        Stiffness tensor, shape ``(3, 3, 3, 3)``. It must have minor and
        major symmetry.

    Returns
    -------
    jnp.ndarray
        Voigt stiffness matrix, shape ``(6, 6)``.

    Raises
    ------
    ValueError
        If the tensor shape is wrong or minor or major symmetry is violated.
    """
    C_np = np.asarray(C_tensor)
    if C_np.shape != (3, 3, 3, 3):
        raise ValueError("Stiffness tensor must have shape (3, 3, 3, 3).")
    if not (
        np.allclose(C_np, C_np.transpose(1, 0, 2, 3))
        and np.allclose(C_np, C_np.transpose(0, 1, 3, 2))
    ):
        raise ValueError("Stiffness tensor must have minor symmetry.")
    if not np.allclose(C_np, C_np.transpose(2, 3, 0, 1)):
        raise ValueError("Stiffness tensor must have major symmetry.")
    C_tensor = jnp.asarray(C_tensor)
    ii, jj = _VOIGT_PAIRS_I, _VOIGT_PAIRS_J
    return C_tensor[ii[:, None], jj[:, None], ii[None, :], jj[None, :]]


def strain_voigt_to_tensor(e):
    """
    Convert engineering Voigt strain to the symmetric 3x3 strain tensor.

    Parameters
    ----------
    e : jnp.ndarray
        Engineering Voigt strain ``(e_xx, e_yy, e_zz, 2e_yz, 2e_xz, 2e_xy)``,
        shape ``(6,)``.

    Returns
    -------
    jnp.ndarray
        Symmetric strain tensor ``eps_ij``, shape ``(3, 3)``.
    """
    e = jnp.asarray(e)
    if e.shape != (6,):
        raise ValueError("Voigt strain must have shape (6,).")
    exx, eyy, ezz, gyz, gxz, gxy = e[0], e[1], e[2], e[3], e[4], e[5]
    return jnp.stack(
        (
            jnp.stack((exx, gxy / 2.0, gxz / 2.0)),
            jnp.stack((gxy / 2.0, eyy, gyz / 2.0)),
            jnp.stack((gxz / 2.0, gyz / 2.0, ezz)),
        )
    )


def homo_elastic_energy_general(global_strain, parameters):
    """
    Return the homogeneous elastic energy for an arbitrary-symmetry stiffness.

    The energy is ``E = 1/2 V_total * eps_ij C_ijkl eps_kl``, summed with
    ``einsum``. A cubic tensor with ``V_total * C11 = N * B11`` (and likewise
    for B12 and B44) reproduces :func:`openferro.engine.elastic.homo_elastic_energy`.

    Parameters
    ----------
    global_strain : jnp.ndarray
        Global engineering-Voigt strain, shape ``(6,)``.
    parameters : tuple
        ``(C_tensor, V_total)``, where ``C_tensor`` has shape ``(3, 3, 3, 3)``
        in eV/Angstrom^3 and ``V_total`` is the supercell volume in Angstrom^3.

    Returns
    -------
    jnp.ndarray
        The homogeneous elastic energy in eV.
    """
    C_tensor, V_total = parameters
    eps = strain_voigt_to_tensor(global_strain)
    return 0.5 * V_total * jnp.einsum("ij,ijkl,kl->", eps, C_tensor, eps)


def _one_sided_difference(values, axis, sign):
    """
    Return the one-sided lattice difference of ``values`` along ``axis``.

    ``sign=+1`` gives ``v(R+b) - v(R)`` and ``sign=-1`` gives
    ``v(R) - v(R-b)``. Periodic boundaries are applied with ``jnp.roll``.
    """
    if sign > 0:
        return jnp.roll(values, -1, axis=axis) - values
    return values - jnp.roll(values, 1, axis=axis)


def inhomo_elastic_energy_general(local_displacement, parameters):
    """
    Return the inhomogeneous elastic energy for an arbitrary-symmetry stiffness.

    The field ``v_a(R) = u_a(R) / a_a`` is the displacement along axis ``a``
    in units of the lattice constant ``a_a``. The displacement gradient along
    lattice axis ``b`` is approximated by a one-sided difference:

    ``G_ab = (a_a / a_b) * D^{s_b}_b v_a``

    with the forward difference ``D^+ v = jnp.roll(v, -1, axis=b) - v``
    (``v(R+b) - v(R)``) and the backward difference
    ``D^- v = v - jnp.roll(v, 1, axis=b)`` (``v(R) - v(R-b)``). The strain is
    ``eps = sym(G)``, and the energy is ``1/2 V_cell eps C eps`` summed over
    all cells.

    The loop runs over the 8 sign patterns ``itertools.product((1, -1),
    repeat=3)``. Each pattern picks a forward or backward ``jnp.roll``
    difference along x, y and z, builds the gradient, symmetrizes it into a
    strain, and computes the energy with ``einsum``. The 8 energies are then
    averaged.

    A single choice already has no zero-energy modes beyond the three uniform
    shifts on a periodic lattice. It is not mirror symmetric, however: the
    energy changes under x -> -x, y -> -y and z -> -z. For example, a transverse
    wave along [110] with polarization [1-10] and a wave along [1-10] with
    polarization [110] have the same energy in the continuum, and both agree
    with it to ``O((k a)^2)``, but a single choice gives them different
    energies at finite k. Averaging over the 8 choices restores the mirror
    symmetry, so these waves cost the same energy. This matters for domain
    walls: (110) and (1-10) twin walls must be equivalent.

    Parameters
    ----------
    local_displacement : jnp.ndarray
        Local displacement field ``v_a(R) = u_a / a_a``, shape
        ``(l1, l2, l3, 3)``.
    parameters : tuple
        ``(C_tensor, lattice_constants, V_cell)``, where ``C_tensor`` has
        shape ``(3, 3, 3, 3)`` in eV/Angstrom^3, ``lattice_constants`` is
        ``(a1, a2, a3)`` in Angstrom for an orthogonal lattice, and ``V_cell``
        is the primitive-cell volume in Angstrom^3.

    Returns
    -------
    jnp.ndarray
        The inhomogeneous elastic energy in eV, summed over the lattice.
    """
    C_tensor, lattice_constants, V_cell = parameters
    lattice_constants = jnp.asarray(lattice_constants)
    # ratio[a, b] = a_a / a_b converts D_b v_a (lattice units) to du_a/dx_b.
    ratio = lattice_constants[:, None] / lattice_constants[None, :]

    energy = 0.0
    for signs in itertools.product((1, -1), repeat=3):
        # Stack b = x, y, z along the last axis: diff[..., a, b] = D^{s_b}_b v_a.
        diff = jnp.stack(
            [
                _one_sided_difference(local_displacement, axis=b, sign=signs[b])
                for b in range(3)
            ],
            axis=-1,
        )
        grad = diff * ratio
        eps = 0.5 * (grad + jnp.swapaxes(grad, -1, -2))
        energy = energy + 0.5 * V_cell * jnp.einsum(
            "...ij,ijkl,...kl->", eps, C_tensor, eps
        )
    return energy / 8.0
