"""Trait closure and propagation rules."""

from pygeoinf2.traits import (
    Traits,
    adjoint_traits,
    close,
    compose_traits,
    congruence_traits,
    gramian_traits,
    inverse_traits,
    scale_traits,
    sum_traits,
)

T = Traits


def has(traits, member):
    return traits & member == member


class TestClosure:
    def test_positive_definite_implies_the_rest(self):
        t = close(T.POSITIVE_DEFINITE)
        assert has(t, T.POSITIVE_SEMIDEFINITE)
        assert has(t, T.INVERTIBLE)
        assert has(t, T.SELF_ADJOINT)

    def test_semidefinite_implies_self_adjoint(self):
        assert has(close(T.POSITIVE_SEMIDEFINITE), T.SELF_ADJOINT)

    def test_unitary_implies_isometry_and_invertible(self):
        t = close(T.UNITARY)
        assert has(t, T.ISOMETRY) and has(t, T.INVERTIBLE)

    def test_orthogonal_projection_is_semidefinite(self):
        assert has(close(T.IDEMPOTENT | T.SELF_ADJOINT), T.POSITIVE_SEMIDEFINITE)

    def test_invertible_semidefinite_is_definite(self):
        assert has(close(T.POSITIVE_SEMIDEFINITE | T.INVERTIBLE), T.POSITIVE_DEFINITE)

    def test_invertible_isometry_is_unitary(self):
        assert has(close(T.ISOMETRY | T.INVERTIBLE), T.UNITARY)

    def test_closure_is_idempotent(self):
        for t in (
            T.NONE,
            T.POSITIVE_DEFINITE,
            T.UNITARY,
            T.IDEMPOTENT | T.SELF_ADJOINT,
        ):
            assert close(close(t)) == close(t)


class TestSum:
    def test_self_adjointness_needs_both(self):
        assert has(sum_traits(T.SELF_ADJOINT, T.SELF_ADJOINT), T.SELF_ADJOINT)
        assert not has(sum_traits(T.SELF_ADJOINT, T.NONE), T.SELF_ADJOINT)

    def test_semidefinite_plus_semidefinite(self):
        t = sum_traits(T.POSITIVE_SEMIDEFINITE, T.POSITIVE_SEMIDEFINITE)
        assert has(t, T.POSITIVE_SEMIDEFINITE)
        assert not has(t, T.POSITIVE_DEFINITE)

    def test_definite_plus_semidefinite_is_definite(self):
        """The normal operator A Q A* + R, with R only semidefinite."""
        t = sum_traits(T.POSITIVE_DEFINITE, T.POSITIVE_SEMIDEFINITE)
        assert has(t, T.POSITIVE_DEFINITE)
        assert has(
            sum_traits(T.POSITIVE_SEMIDEFINITE, T.POSITIVE_DEFINITE),
            T.POSITIVE_DEFINITE,
        )

    def test_definite_plus_nothing_keeps_nothing(self):
        assert sum_traits(T.POSITIVE_DEFINITE, T.NONE) == T.NONE


class TestScale:
    def test_positive_scaling_preserves_definiteness(self):
        t = scale_traits(T.POSITIVE_DEFINITE, 2.5, square=True)
        assert has(t, T.POSITIVE_DEFINITE)

    def test_negative_scaling_drops_definiteness_but_keeps_symmetry(self):
        t = scale_traits(T.POSITIVE_DEFINITE, -2.5, square=True)
        assert has(t, T.SELF_ADJOINT)
        assert has(t, T.INVERTIBLE)
        assert not has(t, T.POSITIVE_SEMIDEFINITE)

    def test_zero_scaling_is_the_zero_operator(self):
        assert has(
            scale_traits(T.POSITIVE_DEFINITE, 0.0, square=True), T.POSITIVE_SEMIDEFINITE
        )
        assert not has(
            scale_traits(T.POSITIVE_DEFINITE, 0.0, square=True), T.INVERTIBLE
        )
        assert scale_traits(T.SELF_ADJOINT, 0.0, square=False) == T.NONE

    def test_scaling_breaks_idempotency_unless_trivial(self):
        assert has(scale_traits(T.IDEMPOTENT, 1.0, square=True), T.IDEMPOTENT)
        assert not has(scale_traits(T.IDEMPOTENT, 2.0, square=True), T.IDEMPOTENT)

    def test_unit_scaling_preserves_isometry(self):
        assert has(scale_traits(T.ISOMETRY, -1.0, square=False), T.ISOMETRY)
        assert not has(scale_traits(T.ISOMETRY, 2.0, square=False), T.ISOMETRY)


class TestAdjoint:
    def test_isometry_is_not_preserved(self):
        """The adjoint of an isometry is a co-isometry."""
        assert not has(adjoint_traits(T.ISOMETRY), T.ISOMETRY)

    def test_unitary_is_preserved(self):
        assert has(adjoint_traits(T.UNITARY), T.UNITARY)
        assert has(adjoint_traits(T.UNITARY), T.ISOMETRY)

    def test_definiteness_and_symmetry_survive(self):
        t = adjoint_traits(T.POSITIVE_DEFINITE)
        assert has(t, T.POSITIVE_DEFINITE) and has(t, T.SELF_ADJOINT)

    def test_adjoint_is_an_involution_on_traits(self):
        for t in (T.POSITIVE_DEFINITE, T.UNITARY, T.SELF_ADJOINT, T.IDEMPOTENT):
            assert adjoint_traits(adjoint_traits(close(t))) == adjoint_traits(close(t))


class TestCompose:
    def test_isometries_compose(self):
        assert has(compose_traits(T.ISOMETRY, T.ISOMETRY, square=False), T.ISOMETRY)

    def test_invertibility_needs_squareness(self):
        assert has(
            compose_traits(T.INVERTIBLE, T.INVERTIBLE, square=True), T.INVERTIBLE
        )
        assert not has(
            compose_traits(T.INVERTIBLE, T.INVERTIBLE, square=False), T.INVERTIBLE
        )

    def test_symmetry_is_not_inherited_by_a_general_product(self):
        """A B is not self-adjoint just because A and B are."""
        assert not has(
            compose_traits(T.SELF_ADJOINT, T.SELF_ADJOINT, square=True), T.SELF_ADJOINT
        )


class TestInverse:
    def test_definiteness_survives(self):
        assert has(inverse_traits(T.POSITIVE_DEFINITE), T.POSITIVE_DEFINITE)

    def test_bare_semidefiniteness_does_not(self):
        assert not has(inverse_traits(T.POSITIVE_SEMIDEFINITE), T.POSITIVE_SEMIDEFINITE)

    def test_result_is_invertible(self):
        assert has(inverse_traits(T.SELF_ADJOINT), T.INVERTIBLE)


class TestStructuralPatterns:
    def test_gramian_is_semidefinite(self):
        t = gramian_traits(invertible=False)
        assert has(t, T.SELF_ADJOINT) and has(t, T.POSITIVE_SEMIDEFINITE)
        assert not has(t, T.POSITIVE_DEFINITE)

    def test_gramian_of_an_invertible_factor_is_definite(self):
        assert has(gramian_traits(invertible=True), T.POSITIVE_DEFINITE)

    def test_congruence_preserves_semidefiniteness(self):
        """The covariance pushforward A Q A*."""
        t = congruence_traits(T.POSITIVE_SEMIDEFINITE, outer_invertible=False)
        assert has(t, T.POSITIVE_SEMIDEFINITE)

    def test_congruence_needs_an_invertible_flank_for_definiteness(self):
        assert not has(
            congruence_traits(T.POSITIVE_DEFINITE, outer_invertible=False),
            T.POSITIVE_DEFINITE,
        )
        assert has(
            congruence_traits(T.POSITIVE_DEFINITE, outer_invertible=True),
            T.POSITIVE_DEFINITE,
        )

    def test_congruence_of_an_unstructured_operator_gives_nothing(self):
        assert congruence_traits(T.NONE, outer_invertible=True) == T.NONE


def test_bayesian_normal_operator_is_recognised():
    """A Q A* + R, the operator CG is asked to invert throughout the library."""
    pushforward = congruence_traits(T.POSITIVE_SEMIDEFINITE, outer_invertible=False)
    noise = close(T.POSITIVE_DEFINITE)
    normal = sum_traits(pushforward, noise)
    assert has(normal, T.SELF_ADJOINT)
    assert has(normal, T.POSITIVE_DEFINITE)
