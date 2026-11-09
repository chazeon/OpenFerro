import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from openferro.engine.elastic import homo_elastic_energy, inhomo_elastic_energy  # noqa: E402
from openferro.engine.elastic_general import (  # noqa: E402
    GPA_PER_EV_PER_ANGSTROM3,
    homo_elastic_energy_general,
    inhomo_elastic_energy_general,
    strain_voigt_to_tensor,
    tensor_to_voigt,
    voigt_to_tensor,
)
from openferro.lattice import BravaisLattice3D  # noqa: E402
from openferro.system import System  # noqa: E402


GPA = GPA_PER_EV_PER_ANGSTROM3

# Orthorhombic stiffness in GPa, nine independent constants.
ORTHO_GPA = np.array(
    [
        [600.0, 250.0, 200.0, 0.0, 0.0, 0.0],
        [250.0, 650.0, 220.0, 0.0, 0.0, 0.0],
        [200.0, 220.0, 900.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 250.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 270.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 300.0],
    ]
)
ORTHO_EV = ORTHO_GPA / GPA

# Orthogonal lattice with a != b != c (tetragonal-like Sv numbers, scaled).
ORTHO_LATTICE = (4.0, 4.6, 2.8)

CUBIC_GPA = np.array(
    [
        [600.0, 250.0, 250.0, 0.0, 0.0, 0.0],
        [250.0, 600.0, 250.0, 0.0, 0.0, 0.0],
        [250.0, 250.0, 600.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 250.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 250.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 250.0],
    ]
)
CUBIC_EV = CUBIC_GPA / GPA
CUBIC_LATTICE = (4.0, 4.0, 4.0)


def _random_symmetric_voigt(rng):
    A = rng.normal(size=(6, 6))
    return A + A.T


def _continuum_energy(C_voigt_ev, lattice, shape, m, polarization):
    """Continuum energy of u(R) = U0 cos(k.R) sampled on the lattice.

    ``k_b = 2 pi m_b / (L_b a_b)`` makes the wave periodic on the box. The
    strain is evaluated analytically at the lattice points.
    """
    a = np.asarray(lattice, dtype=float)
    k = 2 * np.pi * np.asarray(m, dtype=float) / (np.asarray(shape) * a)
    cells = np.indices(shape).reshape(3, -1).T
    positions = cells * a
    sin_phase = np.sin(positions @ k)
    kU = np.outer(polarization, k)
    eps = -0.5 * (kU + kU.T)[None] * sin_phase[:, None, None]
    C_tensor = np.asarray(voigt_to_tensor(C_voigt_ev))
    V_cell = np.prod(a)
    return 0.5 * V_cell * np.einsum("nij,ijkl,nkl->", eps, C_tensor, eps)


def _cosine_field(lattice, shape, m, polarization):
    """Local displacement field v_a = u_a / a_a for u(R) = U0 cos(k.R)."""
    a = np.asarray(lattice, dtype=float)
    k = 2 * np.pi * np.asarray(m, dtype=float) / (np.asarray(shape) * a)
    cells = np.indices(shape).transpose(1, 2, 3, 0)
    phase = (cells * a) @ k
    u = np.asarray(polarization, dtype=float) * np.cos(phase)[..., None]
    return jnp.asarray(u / a)


def _wave_shape(m, L):
    """Grid with L sites along each wave axis and 8 sites elsewhere."""
    return tuple(L if mb != 0 else 8 for mb in m)


def _inhomo_parameters(C_voigt_ev, lattice):
    a = np.asarray(lattice, dtype=float)
    return (
        voigt_to_tensor(C_voigt_ev),
        jnp.asarray(a),
        jnp.asarray(np.prod(a)),
    )


def _discrete_over_continuum(C_voigt_ev, lattice, shape, m, polarization):
    field = _cosine_field(lattice, shape, m, polarization)
    parameters = _inhomo_parameters(C_voigt_ev, lattice)
    energy = float(inhomo_elastic_energy_general(field, parameters))
    reference = _continuum_energy(C_voigt_ev, lattice, shape, m, polarization)
    return energy / reference - 1.0


def test_voigt_tensor_round_trip_and_symmetry():
    rng = np.random.default_rng(0)
    C_voigt = _random_symmetric_voigt(rng)
    C_tensor = np.asarray(voigt_to_tensor(C_voigt))

    # Minor symmetry: C_ijkl = C_jikl = C_ijlk. Major: C_ijkl = C_klij.
    np.testing.assert_allclose(C_tensor, C_tensor.transpose(1, 0, 2, 3))
    np.testing.assert_allclose(C_tensor, C_tensor.transpose(0, 1, 3, 2))
    np.testing.assert_allclose(C_tensor, C_tensor.transpose(2, 3, 0, 1))

    np.testing.assert_allclose(np.asarray(tensor_to_voigt(C_tensor)), C_voigt)
    # Standard map: C_voigt[3,3] (i.e. C44) is C[1,2,1,2].
    np.testing.assert_allclose(C_tensor[1, 2, 1, 2], C_voigt[3, 3])
    np.testing.assert_allclose(C_tensor[0, 2, 0, 1], C_voigt[4, 5])


def test_voigt_helpers_reject_bad_input():
    rng = np.random.default_rng(1)
    C_voigt = _random_symmetric_voigt(rng)
    with pytest.raises(ValueError, match="symmetric"):
        voigt_to_tensor(C_voigt + np.triu(np.ones((6, 6)), 1))
    with pytest.raises(ValueError, match=r"\(6, 6\)"):
        voigt_to_tensor(np.zeros((3, 3)))
    bad_tensor = np.asarray(voigt_to_tensor(C_voigt)).copy()
    bad_tensor[0, 1, 0, 2] += 1.0
    with pytest.raises(ValueError, match="minor symmetry"):
        tensor_to_voigt(bad_tensor)


def test_strain_voigt_to_tensor_uses_engineering_shear():
    e = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expected = np.array(
        [
            [1.0, 3.0, 2.5],
            [3.0, 2.0, 2.0],
            [2.5, 2.0, 3.0],
        ]
    )
    np.testing.assert_allclose(np.asarray(strain_voigt_to_tensor(e)), expected)


def test_homogeneous_general_matches_cubic_energy():
    rng = np.random.default_rng(2)
    B11, B12, B44 = 12.0, 3.0, 8.0
    n_cells = 5
    V_cell = 64.0
    V_total = n_cells * V_cell
    # Cubic tensor with V_total * C = N * B, i.e. C = B / V_cell.
    C_voigt = np.zeros((6, 6))
    C_voigt[:3, :3] = B12 / V_cell
    np.fill_diagonal(C_voigt[:3, :3], B11 / V_cell)
    C_voigt[3:, 3:] = np.diag([B44 / V_cell] * 3)
    C_tensor = voigt_to_tensor(C_voigt)

    cubic_parameters = jnp.array([B11, B12, B44, n_cells])
    for _ in range(5):
        strain = jnp.asarray(rng.normal(scale=0.05, size=6))
        np.testing.assert_allclose(
            homo_elastic_energy_general(strain, (C_tensor, V_total)),
            homo_elastic_energy(strain, cubic_parameters),
            rtol=1e-12,
        )


def test_homogeneous_general_orthorhombic_uniaxial_and_shear():
    C_tensor = voigt_to_tensor(ORTHO_EV)
    V_total = 320.0
    uniaxial = jnp.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        homo_elastic_energy_general(uniaxial, (C_tensor, V_total)),
        0.5 * V_total * ORTHO_EV[0, 0] * 0.02**2,
    )
    shear = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.03])
    np.testing.assert_allclose(
        homo_elastic_energy_general(shear, (C_tensor, V_total)),
        0.5 * V_total * ORTHO_EV[5, 5] * 0.03**2,
    )


@pytest.mark.parametrize(
    "m, polarization",
    [
        ((1, 0, 0), (1, 0, 0)),  # longitudinal along a
        ((1, 0, 0), (0, 1, 0)),  # transverse along a
        ((0, 1, 0), (0, 1, 0)),  # longitudinal along b
        ((0, 0, 1), (1, 0, 0)),  # transverse along c
        ((1, 0, 1), (1, 0, 0)),  # oblique wave
        ((1, 0, 1), (1, 0, 1)),  # oblique, longitudinal
        ((1, 0, 1), (1, 0, -1)),  # oblique, transverse
    ],
)
def test_inhomogeneous_general_long_wavelength_orthorhombic(m, polarization):
    polarization = np.asarray(polarization, dtype=float)
    errors = []
    for L in (16, 32, 64):
        shape = _wave_shape(m, L)
        errors.append(
            _discrete_over_continuum(ORTHO_EV, ORTHO_LATTICE, shape, m, polarization)
        )
    errors = np.abs(errors)
    # The error is O((k a)^2), so doubling the wavelength count should cut it
    # by about 4.
    assert errors[1] / errors[0] == pytest.approx(0.25, rel=0.15)
    assert errors[2] / errors[1] == pytest.approx(0.25, rel=0.15)
    assert errors[2] < 2e-3


def test_inhomogeneous_general_translation_is_free():
    shape = (4, 4, 4)
    uniform = jnp.broadcast_to(
        jnp.array([0.3, -0.2, 0.1]) / jnp.asarray(ORTHO_LATTICE), shape + (3,)
    )
    energy = inhomo_elastic_energy_general(
        uniform, _inhomo_parameters(ORTHO_EV, ORTHO_LATTICE)
    )
    assert abs(float(energy)) < 1e-20


def test_inhomogeneous_general_checkerboard_costs_energy():
    """Sanity check: the averaged engine charges a checkerboard displacement."""
    shape = (4, 4, 4)
    parity = (-1.0) ** np.indices(shape).sum(axis=0)
    checkerboard = np.zeros(shape + (3,))
    checkerboard[..., 0] = parity / np.asarray(ORTHO_LATTICE)[0]
    energy = float(
        inhomo_elastic_energy_general(
            jnp.asarray(checkerboard), _inhomo_parameters(ORTHO_EV, ORTHO_LATTICE)
        )
    )
    assert energy > 0.0
    # Shear-coupled checkerboards also carry energy.
    checkerboard_y = np.zeros(shape + (3,))
    checkerboard_y[..., 1] = parity / np.asarray(ORTHO_LATTICE)[1]
    assert (
        float(
            inhomo_elastic_energy_general(
                jnp.asarray(checkerboard_y),
                _inhomo_parameters(ORTHO_EV, ORTHO_LATTICE),
            )
        )
        > 0.0
    )


def test_mirror_symmetry_110_vs_1m10():
    """Transverse waves along [110] and [1-10] cost the same energy.

    The 8-choice average is invariant under the mirror x -> -x, which maps
    k=(1,1,0), U=(1,-1,0) onto k=(1,-1,0), U=(1,1,0). A single forward-only
    choice is not mirror symmetric and gives a different energy here.
    """
    # Tetragonal-compatible orthorhombic C (C11=C22, C13=C23, C44=C55), in GPa.
    tetra_gpa = np.array(
        [
            [600.0, 250.0, 200.0, 0.0, 0.0, 0.0],
            [250.0, 600.0, 200.0, 0.0, 0.0, 0.0],
            [200.0, 200.0, 900.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 270.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 270.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 300.0],
        ]
    )
    C_ev = tetra_gpa / GPA
    lattice = (1.0, 1.0, 1.5)  # a = b
    shape = (8, 8, 2)  # k = 2 pi / 8 along x and y

    wave_110 = _cosine_field(lattice, shape, (1, 1, 0), np.array([1.0, -1.0, 0.0]))
    wave_1m10 = _cosine_field(lattice, shape, (1, -1, 0), np.array([1.0, 1.0, 0.0]))
    parameters = _inhomo_parameters(C_ev, lattice)

    e_110 = float(inhomo_elastic_energy_general(wave_110, parameters))
    e_1m10 = float(inhomo_elastic_energy_general(wave_1m10, parameters))
    assert e_110 > 0.0
    assert e_110 == pytest.approx(e_1m10, rel=1e-10)

    # Contrast: one forward-only choice breaks the mirror symmetry.
    C_tensor, lattice_constants, V_cell = parameters
    ratio = lattice_constants[:, None] / lattice_constants[None, :]

    def forward_only_energy(v):
        diff = jnp.stack(
            [jnp.roll(v, -1, axis=b) - v for b in range(3)], axis=-1
        )
        grad = diff * ratio
        eps = 0.5 * (grad + jnp.swapaxes(grad, -1, -2))
        return 0.5 * V_cell * jnp.einsum("...ij,ijkl,...kl->", eps, C_tensor, eps)

    f_110 = float(forward_only_energy(wave_110))
    f_1m10 = float(forward_only_energy(wave_1m10))
    assert abs(f_110 / f_1m10 - 1.0) > 0.1


def test_old_cubic_inhomogeneous_engine_against_continuum_report():
    """Report only: how the existing cubic engine compares with continuum.

    The cubic engine ``inhomo_elastic_energy`` is not changed. This prints
    E_old / E_continuum for three plane waves with cubic CUBIC_EV and
    a = 4 Angstrom. It makes no assertion.
    """
    cases = [
        ("longitudinal k||[100], U||[100] (C11)", (1, 0, 0), (1, 0, 0)),
        ("transverse   k||[100], U||[010] (C44)", (1, 0, 0), (0, 1, 0)),
        ("longitudinal k||[110], U||[110] (C12)", (1, 1, 0), (1, 1, 0)),
        ("transverse   k||[110], U||[1-10] (C44 mixed)", (1, 1, 0), (1, -1, 0)),
    ]
    C = CUBIC_EV
    V_cell = CUBIC_LATTICE[0] ** 3
    B11, B12, B44 = C[0, 0] * V_cell, C[0, 1] * V_cell, C[3, 3] * V_cell
    old_parameters = jnp.array([B11, B12, B44])

    lines = [
        "",
        "Old cubic inhomo_elastic_energy vs continuum (a = 4 A, cubic C)",
        f"{'case':48s} {'L':>4s} {'E_old/E_cont':>14s} {'E_new/E_cont':>14s}",
    ]
    for name, m, polarization in cases:
        polarization = np.asarray(polarization, dtype=float)
        for L in (16, 32, 64):
            shape = _wave_shape(m, L)
            field = _cosine_field(CUBIC_LATTICE, shape, m, polarization)
            reference = _continuum_energy(C, CUBIC_LATTICE, shape, m, polarization)
            e_old = float(inhomo_elastic_energy(field, old_parameters))
            e_new = float(
                inhomo_elastic_energy_general(
                    field, _inhomo_parameters(C, CUBIC_LATTICE)
                )
            )
            lines.append(
                f"{name:48s} {L:4d} {e_old / reference:14.5f} {e_new / reference:14.5f}"
            )
    print("\n".join(lines))


def test_system_general_interactions_use_gpa_and_lattice():
    lattice = BravaisLattice3D(
        4,
        3,
        2,
        a1=jnp.array([4.0, 0.0, 0.0]),
        a2=jnp.array([0.0, 4.6, 0.0]),
        a3=jnp.array([0.0, 0.0, 2.8]),
    )
    system = System(lattice)
    strain = system.add_global_strain(value=jnp.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    local = system.add_field("lstrain", ftype="LocalStrain3D")
    displacement = _cosine_field(
        ORTHO_LATTICE, (4, 3, 2), (1, 0, 0), np.array([1.0, 0.0, 0.0])
    )
    local.set_values(displacement)

    homo = system.add_homo_elastic_general_interaction(
        "homo", "gstrain", ORTHO_GPA, unit="GPa"
    )
    inhomo = system.add_inhomo_elastic_general_interaction(
        "inhomo", "lstrain", ORTHO_GPA, unit="GPa"
    )

    V_total = 24 * lattice.unit_volume
    expected_homo = homo_elastic_energy_general(
        strain.get_values(), (voigt_to_tensor(ORTHO_EV), V_total)
    )
    np.testing.assert_allclose(
        float(homo.calc_energy(strain)), float(expected_homo), rtol=1e-12
    )

    expected_inhomo = inhomo_elastic_energy_general(
        displacement, _inhomo_parameters(ORTHO_EV, ORTHO_LATTICE)
    )
    np.testing.assert_allclose(
        float(inhomo.calc_energy(local)), float(expected_inhomo), rtol=1e-12
    )

    # eV/A^3 input must give the same result as the GPa input.
    same = system.add_homo_elastic_general_interaction(
        "homo_ev", "gstrain", ORTHO_EV, unit="eV/A^3"
    )
    np.testing.assert_allclose(
        float(same.calc_energy(strain)), float(homo.calc_energy(strain)), rtol=1e-12
    )


def test_system_general_interactions_validate_input():
    lattice = BravaisLattice3D(2, 2, 2, a1=jnp.array([4.0, 0.0, 0.0]))
    system = System(lattice)
    system.add_global_strain(value=jnp.zeros(6))
    system.add_field("lstrain", ftype="LocalStrain3D")

    with pytest.raises(ValueError, match=r"\(6, 6\)"):
        system.add_homo_elastic_general_interaction("bad", "gstrain", np.zeros((3, 3)))
    asymmetric = ORTHO_GPA.copy()
    asymmetric[0, 1] += 10.0
    with pytest.raises(ValueError, match="symmetric"):
        system.add_inhomo_elastic_general_interaction("bad", "lstrain", asymmetric, unit="GPa")
    with pytest.raises(ValueError, match="unit"):
        system.add_homo_elastic_general_interaction(
            "bad", "gstrain", ORTHO_GPA, unit="Mbar"
        )

    not_pd = np.eye(6)
    not_pd[0, 0] = -1.0
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        system.add_homo_elastic_general_interaction("npd", "gstrain", not_pd)
    assert any("positive definite" in str(w.message) for w in record)
