"""
test_data_loading.py — Verify data loading from the actual data/ folder.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loading.prepare_data import discover_subjects
from data_loading.mesh_to_volume import load_stl, combine_fragments, center_and_normalize, stl_to_sdf


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_dir = os.path.abspath(data_dir)

    print("=" * 60)
    print("STEP 1: SUBJECT DISCOVERY")
    print("=" * 60)
    subjects = discover_subjects(data_dir)
    print()

    # Tally by type
    types = {}
    for s in subjects:
        t = s['subject_type']
        types[t] = types.get(t, 0) + 1
    for t, c in sorted(types.items()):
        print(f"  {t}: {c} subjects")
    print()

    print("=" * 60)
    print("STEP 2: STL LOADING (all subjects)")
    print("=" * 60)
    load_ok = 0
    load_fail = 0
    for s in subjects:
        sid = s['subject_id']
        for f in s['stl_files']:
            fname = os.path.basename(f)
            try:
                mesh = load_stl(f)
                verts = len(mesh.vertices)
                faces = len(mesh.faces)
                print(f"  OK   {sid:25s} | {fname:30s} | {verts:6d} verts, {faces:6d} faces")
                load_ok += 1
            except Exception as e:
                print(f"  FAIL {sid:25s} | {fname:30s} | {e}")
                load_fail += 1

    print(f"\nLoaded: {load_ok} OK, {load_fail} FAILED")
    print()

    print("=" * 60)
    print("STEP 3: FRAGMENT COMBINATION (all subjects)")
    print("=" * 60)
    combine_ok = 0
    combine_fail = 0
    for s in subjects:
        sid = s['subject_id']
        try:
            combined = combine_fragments(s['stl_files'])
            normalized, com, scale = center_and_normalize(combined)
            bmin = normalized.bounds[0].round(3)
            bmax = normalized.bounds[1].round(3)
            print(f"  OK   {sid:25s} | {len(combined.vertices):6d} verts | "
                  f"scale={scale:.4f} | bounds=[{bmin} .. {bmax}]")
            combine_ok += 1
        except Exception as e:
            print(f"  FAIL {sid:25s} | {e}")
            combine_fail += 1

    print(f"\nCombined: {combine_ok} OK, {combine_fail} FAILED")
    print()

    print("=" * 60)
    print("STEP 4: FULL SDF PIPELINE (3 subjects, res=32)")
    print("=" * 60)
    test_subjects = subjects[:3]
    sdf_ok = 0
    for s in test_subjects:
        sid = s['subject_id']
        try:
            sdf = stl_to_sdf(s['stl_files'], resolution=32)
            n_inside = int((sdf < 0).sum())
            n_surface = int((np.abs(sdf) < 0.05).sum())
            fill = 100 * n_inside / sdf.size
            print(f"  OK   {sid:25s} | shape={sdf.shape} | "
                  f"range=[{sdf.min():.3f}, {sdf.max():.3f}] | "
                  f"fill={fill:.1f}% | surface_vox={n_surface}")
            sdf_ok += 1
        except Exception as e:
            print(f"  FAIL {sid:25s} | {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Subjects discovered:    {len(subjects)}")
    print(f"  STL files loaded:       {load_ok}/{load_ok + load_fail}")
    print(f"  Fragments combined:     {combine_ok}/{combine_ok + combine_fail}")
    print(f"  SDF conversions (of 3): {sdf_ok}/3")

    all_ok = load_fail == 0 and combine_fail == 0 and sdf_ok == 3
    print()
    if all_ok:
        print("  >>> ALL DATA LOADING CHECKS PASSED! <<<")
    else:
        print("  >>> SOME CHECKS FAILED - SEE ABOVE FOR DETAILS <<<")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
