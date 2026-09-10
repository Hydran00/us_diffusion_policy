import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from us_dp.anatomy import estimate_liver_frame, read_isaac_mesh_to_world
from us_dp.geometry import rotation_matrix


def triangle():
    return np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), np.array([[0, 1, 2]])


def test_exact_surface_centroid_and_moments():
    vertices, faces = triangle()
    frame = estimate_liver_frame(vertices, faces)
    np.testing.assert_allclose(frame.center, [1, 2 / 3, 0], atol=1e-12)
    assert frame.surface_area == pytest.approx(3)
    covariance = frame.axes @ np.diag(frame.variances) @ frame.axes.T
    np.testing.assert_allclose(
        covariance, [[0.5, -1 / 6, 0], [-1 / 6, 2 / 9, 0], [0, 0, 0]], atol=1e-12
    )
    np.testing.assert_allclose(frame.axes.T @ frame.axes, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame.axes) == pytest.approx(1)
    assert np.all(np.diff(frame.variances) <= 0)


def test_subdivision_does_not_bias_pca():
    vertices, faces = triangle()
    original = estimate_liver_frame(vertices, faces)
    # One highly asymmetric interior subdivision creates uneven vertex density.
    vertices = np.vstack((vertices, [0.01, 0.01, 0]))
    split = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
    result = estimate_liver_frame(vertices, split, batch_size=1)
    np.testing.assert_allclose(result.center, original.center, atol=1e-12)
    np.testing.assert_allclose(result.variances, original.variances, atol=1e-12)
    np.testing.assert_allclose(result.axes, original.axes, atol=1e-12)


def test_world_transform_tracks_randomized_phantom():
    vertices, faces = triangle()
    frame = estimate_liver_frame(vertices, faces)
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix([0.2, -0.1, 0.3, 0.9], "xyzw")
    pose[:3, 3] = [0.6, -0.1, 0.09]
    world = frame.in_world(pose)
    np.testing.assert_allclose(world.center, pose[:3, :3] @ frame.center + pose[:3, 3])
    np.testing.assert_allclose(world.axes, pose[:3, :3] @ frame.axes)
    np.testing.assert_array_equal(world.variances, frame.variances)
    transformed = estimate_liver_frame(vertices @ pose[:3, :3].T + pose[:3, 3], faces)
    np.testing.assert_allclose(world.center, transformed.center, atol=1e-7)
    np.testing.assert_allclose(np.abs(world.axes.T @ transformed.axes), np.eye(3), atol=1e-7)


def test_degenerate_geometry_and_invalid_transforms():
    with pytest.raises(ValueError, match="nondegenerate"):
        estimate_liver_frame(np.zeros((3, 3)), [[0, 1, 2]])
    vertices, faces = triangle()
    with pytest.raises(ValueError, match="indices"):
        estimate_liver_frame(vertices, [[0, 1, 8]])
    frame = estimate_liver_frame(vertices, faces)
    pose = np.eye(4)
    pose[0, 0] = 2
    with pytest.raises(ValueError, match="SO"):
        frame.in_world(pose)


def test_isaac_adapter_uses_calibrated_acoustic_mesh_frame(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "i4h_arena.tensor_utils", SimpleNamespace(to_torch=torch.as_tensor)
    )
    data = SimpleNamespace(
        target_pos_w=np.array([[[0.6, 0.02, 0.09]]]),
        target_quat_w=np.array([[[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]]]),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(scene={"mesh_to_organ_transform": SimpleNamespace(data=data)})
    )
    pose = read_isaac_mesh_to_world(env, quaternion_order="xyzw")
    np.testing.assert_allclose(pose[:3, 3], [0.6, 0.02, 0.09])
    np.testing.assert_allclose(pose[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-7)


def test_prepare_anatomy_scales_mm_and_saves_full_resolution_statistics(tmp_path):
    pytest.importorskip("open3d")
    from us_dp.anatomy_viewer import prepare_anatomy

    root = tmp_path / "meshes"
    root.mkdir()
    obj = "v 0 0 0\nv 300 0 0\nv 0 200 0\nf 1 2 3\n"
    for name in ("Skin", "Liver"):
        (root / f"{name}.obj").write_text(obj)
    report = prepare_anatomy(root, tmp_path / "result", display_voxel_m=0.01)
    np.testing.assert_allclose(report["liver_mesh_frame"]["center_m"], [0.1, 0.2 / 3, 0], atol=1e-8)
    assert report["meshes"]["Liver"]["triangles"] == 1
    assert len(report["meshes"]["Liver"]["sha256"]) == 64
    assert (tmp_path / "result/Skin.ply").is_file()
    saved = json.loads((tmp_path / "result/landmarks.json").read_text())
    assert saved["frame"] == "acoustic_mesh"


def test_phantom_axes_are_independent_of_liver_geometry(tmp_path):
    pytest.importorskip("open3d")
    from us_dp.anatomy import LiverFrame, estimate_surface_frame
    from us_dp.anatomy_viewer import prepare_anatomy

    root = tmp_path / "meshes"
    root.mkdir()
    (root / "Liver.obj").write_text("v 0 0 0\nv 300 0 0\nv 0 200 0\nf 1 2 3\n")
    (root / "Skin.obj").write_text("v 0 0 0\nv 0 400 0\nv 0 0 700\nf 1 2 3\n")
    report = prepare_anatomy(root, tmp_path / "result")
    expected = estimate_surface_frame([[0, 0, 0], [0, 0.4, 0], [0, 0, 0.7]], [[0, 1, 2]])
    actual = LiverFrame.from_dict(report["phantom_mesh_frame"])
    np.testing.assert_allclose(actual.axes, expected.axes, atol=1e-8)
    assert not np.allclose(actual.axes, report["liver_mesh_frame"]["axes_columns"])
    np.testing.assert_allclose(
        report["liver_display_frame"]["center_m"], [0.1, 0.2 / 3, 0], atol=1e-8
    )
