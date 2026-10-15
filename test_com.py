"""
Pure-logic self-tests: no MediaPipe, no video. Verifies BSP tables sum to 1
and the CoM math behaves on synthetic keypoints.

Run:  python test_com.py
"""

import numpy as np

from bsp_tables import get_table, list_tables, validate_table
from com import compute_com, SEGMENTS, L


def test_all_tables_sum_to_one():
    for name, sex in list_tables():
        table = get_table(name, sex=sex)   # get_table already validates
        s = validate_table(table)
        assert abs(s - 1.0) < 1e-3, (name, sex, s)
    print("PASS  every BSP table sums to 1.0")


def test_layout_covers_all_mass():
    # The layout in com.py must instantiate exactly the mass the table expects.
    table = get_table("deleva", "male")
    total = sum(table[seg.seg_type].mass for seg in SEGMENTS)
    assert abs(total - 1.0) < 1e-3, total
    print(f"PASS  segment layout covers full body mass ({total:.4f})")


def _synthetic_kps():
    """A crude but valid standing pose in normalized coords (x right, y down)."""
    kps = np.zeros((33, 2))
    kps[L["left_ear"]]  = (0.48, 0.08); kps[L["right_ear"]] = (0.52, 0.08)
    kps[L["left_shoulder"]] = (0.45, 0.22); kps[L["right_shoulder"]] = (0.55, 0.22)
    kps[L["left_elbow"]] = (0.43, 0.38); kps[L["right_elbow"]] = (0.57, 0.38)
    kps[L["left_wrist"]] = (0.42, 0.52); kps[L["right_wrist"]] = (0.58, 0.52)
    kps[L["left_index"]] = (0.42, 0.56); kps[L["right_index"]] = (0.58, 0.56)
    kps[L["left_hip"]] = (0.47, 0.55); kps[L["right_hip"]] = (0.53, 0.55)
    kps[L["left_knee"]] = (0.47, 0.75); kps[L["right_knee"]] = (0.53, 0.75)
    kps[L["left_ankle"]] = (0.47, 0.95); kps[L["right_ankle"]] = (0.53, 0.95)
    kps[L["left_foot_index"]] = (0.45, 0.99); kps[L["right_foot_index"]] = (0.55, 0.99)
    return kps


def test_com_is_symmetric_and_midtrunk():
    kps = _synthetic_kps()
    com, info = compute_com(kps, get_table("deleva", "male"))
    assert com is not None
    # Symmetric pose about x=0.5 -> CoM x should be ~0.5.
    assert abs(com[0] - 0.5) < 1e-6, com
    # CoM should sit in the trunk region (below shoulders 0.22, above hips 0.55),
    # i.e. mid-torso — a well-known sanity range for standing CoM.
    assert 0.30 < com[1] < 0.55, com[1]
    assert abs(info["mass_used"] - 1.0) < 1e-6
    print(f"PASS  CoM symmetric & mid-trunk  (x={com[0]:.3f}, y={com[1]:.3f})")


def test_table_swap_changes_com():
    kps = _synthetic_kps()
    com_dl, _ = compute_com(kps, get_table("deleva", "male"))
    com_dm, _ = compute_com(kps, get_table("dempster"))
    # Different tables -> different (but same-order) CoM height. This is exactly
    # the swappability the sensitivity check relies on.
    assert not np.allclose(com_dl, com_dm), (com_dl, com_dm)
    print(f"PASS  table swap moves CoM  (deLeva y={com_dl[1]:.3f} vs "
          f"Dempster y={com_dm[1]:.3f})")


def test_visibility_dropout_redistributes_mass():
    kps = _synthetic_kps()
    vis = np.ones(33)
    vis[L["left_index"]] = 0.0  # hide one hand endpoint
    com, info = compute_com(kps, get_table("deleva", "male"), visibility=vis)
    assert com is not None
    assert info["mass_used"] < 1.0            # dropped a segment
    assert "hand_L" not in info["segments_used"]
    print(f"PASS  low-visibility segment dropped  (mass_used={info['mass_used']:.3f})")


if __name__ == "__main__":
    test_all_tables_sum_to_one()
    test_layout_covers_all_mass()
    test_com_is_symmetric_and_midtrunk()
    test_table_swap_changes_com()
    test_visibility_dropout_redistributes_mass()
    print("\nAll self-tests passed.")
