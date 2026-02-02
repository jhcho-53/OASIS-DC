# make_list.py
# Usage:
#   # NYUv2 (자동: rgb_da ↔ depth_inpainted_mm, 2컬럼 est gt)
#   python make_list.py --dataset nyuv2 --root /media/vip/T31/NYUv2_official --out nyu_val.txt
#
#   # 3컬럼(est sparse gt)로 만들고 싶다면(NUY엔 sparse가 없어서 placeholder 사용)
#   python make_list.py --dataset nyuv2 --root /media/vip/T31/NYUv2_official --out nyu_triplet.txt \
#       --nyu-format est-sparse-gt --nyu-sparse-placeholder -
#
#   # 디렉터리를 자동이 아닌 수동으로 지정
#   python make_list.py --dataset nyuv2 --root /media/vip/T31/NYUv2_official --out nyu_val.txt \
#       --nyu-est-dir /media/vip/T31/NYUv2_official/rgb_da \
#       --nyu-gt-dir  /media/vip/T31/NYUv2_official/depth_inpainted_mm
#
#   # (참고) KITTI도 유지. 필요 시 사용하세요.
#   python make_list.py --dataset kitti --root /data/kitti --out kitti_val.txt
#
import os, glob, argparse
from collections import defaultdict

def _abs(p): return os.path.abspath(p)

def _ensure_parent_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

def _write_lines(lines, out_path):
    _ensure_parent_dir(out_path)
    with open(out_path, "w") as f:
        for ln in lines: f.write(ln + "\n")
    print(f"[OK] wrote {len(lines)} lines -> {out_path}")

def _index_by_stem(dir_path, exts):
    idx = defaultdict(list)
    for ext in exts:
        for p in glob.glob(os.path.join(dir_path, "**", f"*{ext}"), recursive=True):
            idx[os.path.splitext(os.path.basename(p))[0]].append(_abs(p))
    uniq, collisions = {}, 0
    for stem, paths in idx.items():
        if len(paths) == 1: uniq[stem] = paths[0]
        else: collisions += 1
    if collisions:
        print(f"[WARN] collisions in '{dir_path}': {collisions} stems skipped")
    return uniq

# ---------- NYUv2 ----------
def _autodetect_dir_with_basename(root, basename, exts):
    """
    root 아래에서 basename이 정확히 basename인 디렉터리들 중,
    지정 확장자 파일 수가 가장 많은 디렉터리를 선택
    """
    root = _abs(root)
    cands = []
    for dp, dn, fn in os.walk(root):
        if os.path.basename(dp).lower() == basename.lower():
            n = 0
            for ext in exts:
                n += len(glob.glob(os.path.join(dp, "**", f"*{ext}"), recursive=True))
            if n > 0:
                cands.append((n, _abs(dp)))
    if not cands:
        return None
    cands.sort(reverse=True)
    print(f"[INFO] autodetected '{basename}': {cands[0][1]} (files={cands[0][0]})")
    return cands[0][1]

def make_nyu_list(root, out_path, nyu_est_dir=None, nyu_gt_dir=None,
                  nyu_format="est-gt", nyu_sparse_placeholder="-"):
    """
    nyu_format:
      - 'est-gt'         -> line: <est> <gt>
      - 'est-sparse-gt'  -> line: <est> <sparse> <gt> (NYU엔 sparse가 없으므로 placeholder 사용)
    """
    # 1) est / gt 디렉터리 자동 탐지 (필요시 수동 지정으로 대체)
    if nyu_est_dir is None:
        nyu_est_dir = _autodetect_dir_with_basename(root, "rgb_da", [".png", ".jpg", ".jpeg"])
    if nyu_gt_dir is None:
        nyu_gt_dir  = _autodetect_dir_with_basename(root, "depth_inpainted_mm", [".png"])
    if nyu_est_dir is None or nyu_gt_dir is None:
        raise FileNotFoundError(
            "NYUv2 디렉터리 자동탐지 실패. 다음 폴더명이 root 하위에 있어야 합니다:\n"
            "  - est: 'rgb_da' (png/jpg)\n  - gt : 'depth_inpainted_mm' (png)\n"
            "필요시 --nyu-est-dir / --nyu-gt-dir 로 직접 지정하세요."
        )

    # 2) 파일 인덱스 생성 (basename stem 기준 매칭)
    est_index = _index_by_stem(nyu_est_dir, [".png", ".jpg", ".jpeg"])
    gt_index  = _index_by_stem(nyu_gt_dir,  [".png"])

    common_stems = sorted(set(est_index.keys()) & set(gt_index.keys()))
    if not common_stems:
        raise RuntimeError("매칭되는 파일이 없습니다. (stem 기준) rgb_da와 depth_inpainted_mm의 파일명을 확인하세요.")

    missed_est = len(gt_index) - len(common_stems)
    missed_gt  = len(est_index) - len(common_stems)
    if missed_est > 0: print(f"[WARN] est 미매칭: {missed_est} (gt에는 있으나 est에 없음)")
    if missed_gt  > 0: print(f"[WARN] gt 미매칭: {missed_gt} (est에는 있으나 gt에 없음)")

    # 3) 라인 작성
    lines = []
    if nyu_format == "est-gt":
        for stem in common_stems:
            lines.append(f"{est_index[stem]} {gt_index[stem]}")
    elif nyu_format == "est-sparse-gt":
        for stem in common_stems:
            lines.append(f"{est_index[stem]} {nyu_sparse_placeholder} {gt_index[stem]}")
    else:
        raise ValueError("--nyu-format must be 'est-gt' or 'est-sparse-gt'")

    _write_lines(lines, out_path)

# ---------- KITTI (유지) ----------
def _autodetect_kitti_dirs(root):
    root = _abs(root)
    candidates = [
        ("depth_selection/val_selection_cropped/velodyne_raw",
         "depth_selection/val_selection_cropped/groundtruth_depth"),
        ("depth_selection/val_selection/velodyne_raw",
         "depth_selection/val_selection/groundtruth_depth"),
        ("val/sparse_depth", "val/groundtruth_depth"),
        ("sparse_depth/val", "groundtruth_depth/val"),
    ]
    for sp_rel, gt_rel in candidates:
        sp = os.path.join(root, sp_rel); gt = os.path.join(root, gt_rel)
        if os.path.isdir(sp) and os.path.isdir(gt):
            if glob.glob(os.path.join(sp, "*.png")) and glob.glob(os.path.join(gt, "*.png")):
                return _abs(sp), _abs(gt)
    # fallback: 이름 기반 검색
    velos = [d for d,_,_ in os.walk(root) if d.endswith("velodyne_raw")]
    gts   = [d for d,_,_ in os.walk(root) if d.endswith("groundtruth_depth")]
    for sp in velos:
        for gt in gts:
            if glob.glob(os.path.join(sp, "*.png")) and glob.glob(os.path.join(gt, "*.png")):
                return _abs(sp), _abs(gt)
    raise FileNotFoundError("KITTI val 디렉터리를 찾지 못했습니다. --root 구조를 확인하거나 --sparse-dir/--gt-dir 지정.")

def make_kitti_list(root, out_path, sparse_dir=None, gt_dir=None):
    import glob, os
    if sparse_dir and gt_dir:
        sp_dir, gt_dir = _abs(sparse_dir), _abs(gt_dir)
    else:
        sp_dir, gt_dir = _autodetect_kitti_dirs(root)

    print(f"[KITTI] sparse_dir = {sp_dir}")
    print(f"[KITTI] gt_dir     = {gt_dir}")

    gt_pngs = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    if not gt_pngs:
        raise FileNotFoundError("KITTI GT PNG가 없습니다.")

    lines, missed = [], 0
    for gt in gt_pngs:
        bn = os.path.basename(gt)

        # (1) 동일 파일명 매칭 먼저
        sp = os.path.join(sp_dir, bn)
        if os.path.isfile(sp):
            lines.append(f"{_abs(sp)} {_abs(gt)}")
            continue

        # (2) '..._groundtruth_depth_...' -> '..._velodyne_raw_...' 치환 매칭
        cand = bn.replace("_groundtruth_depth_", "_velodyne_raw_", 1)
        sp2 = os.path.join(sp_dir, cand)
        if os.path.isfile(sp2):
            lines.append(f"{_abs(sp2)} {_abs(gt)}")
            continue

        missed += 1

    if missed:
        print(f"[WARN] sparse 매칭 실패: {missed}")
    _write_lines(lines, out_path)


# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nyuv2","kitti"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    # NYU options
    ap.add_argument("--nyu-est-dir", default=None, help="rgb_da 디렉터리 직접 지정")
    ap.add_argument("--nyu-gt-dir",  default=None, help="depth_inpainted_mm 디렉터리 직접 지정")
    ap.add_argument("--nyu-format",  default="est-gt", choices=["est-gt","est-sparse-gt"])
    ap.add_argument("--nyu-sparse-placeholder", default="-",
                    help="nyu-format=est-sparse-gt일 때 sparse 칼럼에 쓸 placeholder")
    # KITTI options (유지)
    ap.add_argument("--sparse-dir", default=None)
    ap.add_argument("--gt-dir", default=None)
    args = ap.parse_args()

    if args.dataset == "nyuv2":
        make_nyu_list(
            root=args.root, out_path=args.out,
            nyu_est_dir=args.nyu_est_dir, nyu_gt_dir=args.nyu_gt_dir,
            nyu_format=args.nyu_format, nyu_sparse_placeholder=args.nyu_sparse_placeholder
        )
    else:
        make_kitti_list(root=args.root, out_path=args.out,
                        sparse_dir=args.sparse_dir, gt_dir=args.gt_dir)

if __name__ == "__main__":
    main()
