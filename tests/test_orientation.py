"""Geometry of axial orientation: resampling and frame transforms.

Every failure guarded here produces a plausible-looking orientation map rather
than an error, so these are the cheapest insurance in the package.
"""

import numpy as np
import pytest

from polscope_library import (
    average_orientation,
    orientation_to_vector,
    transform_orientation,
    vector_to_orientation,
)


class TestAxialResampling:
    def test_the_trap_averaging_across_the_wrap(self):
        """179 deg and 1 deg are 2 deg apart, not 178. The arithmetic mean
        gives 90 deg -- perpendicular to the truth, and plausible-looking."""
        a = np.deg2rad([179.0, 1.0])
        naive = np.rad2deg(np.mean(a))
        correct = np.rad2deg(average_orientation(a))
        assert np.isclose(naive, 90.0)
        assert np.isclose(correct, 0.0, atol=1e-9) or np.isclose(correct, 180.0, atol=1e-9)

    def test_vector_round_trip(self):
        o = np.deg2rad(np.arange(0, 180, 0.5))
        assert np.allclose(vector_to_orientation(*orientation_to_vector(o)), o, atol=1e-9)

    def test_theta_and_theta_plus_pi_are_the_same_axis(self):
        o = np.deg2rad([10.0, 45.0, 170.0])
        s1, c1 = orientation_to_vector(o)
        s2, c2 = orientation_to_vector(o + np.pi)
        assert np.allclose(s1, s2) and np.allclose(c1, c2)

    def test_weighting_by_retardance_suppresses_undefined_pixels(self):
        """Orientation is undefined where there is no birefringence. Unweighted,
        that noise pulls the mean; weighted by retardance, it does not."""
        rng = np.random.default_rng(0)
        signal = np.full(20, np.deg2rad(30.0))
        noise = rng.uniform(0, np.pi, 200)
        ori = np.concatenate([signal, noise])
        w = np.concatenate([np.full(20, 50.0), np.full(200, 0.01)])
        assert abs(np.rad2deg(average_orientation(ori, weights=w)) - 30.0) < 2.0


class TestFrameTransforms:
    @pytest.mark.parametrize("deg", [0.0, 12.5, 45.0, 90.0, 179.0])
    def test_single_mirror_negates(self, deg):
        o = np.deg2rad(deg)
        for kw in ({"flip_x": True}, {"flip_y": True}):
            got = np.rad2deg(transform_orientation(o, **kw))
            assert np.isclose(got, (-deg) % 180.0, atol=1e-9)

    @pytest.mark.parametrize("deg", [0.0, 12.5, 45.0, 90.0, 179.0])
    def test_double_mirror_is_a_180_rotation_and_leaves_orientation_alone(self, deg):
        """The surprising one. Mirroring both axes composes to a 180 degree
        rotation, and orientation is axial, so the value must NOT change."""
        got = np.rad2deg(transform_orientation(np.deg2rad(deg), flip_x=True, flip_y=True))
        assert np.isclose(got, deg % 180.0, atol=1e-9)

    def test_identity_is_a_no_op(self):
        o = np.deg2rad(np.arange(0, 180, 7.0))
        assert np.allclose(transform_orientation(o), o % np.pi)

    def test_rotation_adds_and_wraps(self):
        assert np.isclose(np.rad2deg(transform_orientation(np.deg2rad(170.0), rotate_deg=30.0)), 20.0)

    def test_result_is_always_in_range(self):
        rng = np.random.default_rng(1)
        o = rng.uniform(-10, 10, 500)
        for kw in ({}, {"flip_x": True}, {"flip_y": True}, {"flip_x": True, "flip_y": True}, {"rotate_deg": -217.0}):
            out = transform_orientation(o, **kw)
            assert out.min() >= 0.0 and out.max() < np.pi

    def test_flip_then_unflip_round_trips(self):
        o = np.deg2rad(np.arange(0, 180, 3.0))
        for kw in ({"flip_x": True}, {"flip_y": True}, {"rotate_deg": 90.0}):
            once = transform_orientation(o, **kw)
            twice = transform_orientation(once, **kw)
            expect = o % np.pi if "rotate_deg" not in kw else (o + np.pi) % np.pi
            assert np.allclose(twice, expect, atol=1e-9)


def test_matches_a_synthetic_radial_target_under_every_transform():
    """End-to-end analogue of the sunburst check: build a radial slow-axis
    field, transform the pixels, and confirm it is still radial about the
    moved centre only when the angles are transformed too."""
    n = 101
    yy, xx = np.mgrid[0:n, 0:n]
    yc = xc = (n - 1) / 2
    ori = np.arctan2(yy - yc, xx - xc) % np.pi  # radial, y-down image frame

    def radial_error(o, cy, cx):
        y, x = np.mgrid[0 : o.shape[0], 0 : o.shape[1]]
        r = np.hypot(y - cy, x - cx)
        m = (r > 5) & (r < 45)
        d = (o[m] - np.arctan2((y - cy)[m], (x - cx)[m])) % np.pi
        return np.rad2deg(np.median(np.minimum(d, np.pi - d)))

    cases = [
        (lambda a: a[:, ::-1], {"flip_x": True}, (yc, n - 1 - xc)),
        (lambda a: a[::-1, :], {"flip_y": True}, (n - 1 - yc, xc)),
        (lambda a: a[::-1, ::-1], {"flip_x": True, "flip_y": True}, (n - 1 - yc, n - 1 - xc)),
        (lambda a: np.rot90(a, 1), {"rotate_deg": 90.0}, (n - 1 - xc, yc)),
    ]
    for move, kw, (cy, cx) in cases:
        assert radial_error(transform_orientation(move(ori), **kw), cy, cx) < 1e-6
