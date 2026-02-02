#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI-DC(train)에서 고정 seed 집합과 shot(1/10/100)별로 프레임을 샘플링한 뒤,
원본과 동일한 상대경로를 유지한 채로 '루트만 교체'한 데이터 뷰를 생성합니다.

출력 구조(예):
  {out}/1shot/S0/<원본 root 이하 상대경로들...>
  {out}/10shot/S0/...
  {out}/100shot/S0/...
  (seed별 S1..S4 동일)

기본은 '심볼릭 링크'로 공간 절약. 필요 시 --link-mode copy|hardlink 가능.

추가 모달리티:
  --pseudo-template, --mono-template 에서 {root},{split},{drive},{frame} 사용
  (반드시 {root} 하위로 경로가 생성되도록 지정해야 상대경로 유지 가능)

사용 예)
python make_kitti_subset_view.py --root /PATH/TO/KITTI_DC \
  --out /PATH/TO/OUTPUT \
  --shots 1 10 100 --seeds 0 1 2 3 4 \
  --pseudo-template "{root}/poisson/{split}/{drive}/image_02/{frame}.png" \
  --mono-template   "{root}/depth_anything/{split}/{drive}/image_02/{frame}.png" \
  --check-extra-exists --link-mode symlink
"""
import os, glob, argparse, shutil, numpy as np

def _exists(p): return p if (p and os.path.exists(p)) else None

def index_kitti(root, split="train"):
    gt_root = os.path.join(root, "data_depth_annotated", split)
    pattern = os.path.join(gt_root, "**", "proj_depth", "groundtruth", "image_02", "*.png")
    gt_files = sorted(glob.glob(pattern, recursive=True))
    items = []
    for g in gt_files:
        parts = os.path.normpath(g).split(os.sep)
        if split not in parts or "proj_depth" not in parts:
            continue
        i_split = parts.index(split)
        i_proj  = parts.index("proj_depth")
        drive = os.path.join(*parts[i_split+1:i_proj])
        frame = os.path.splitext(os.path.basename(g))[0]
        sparse = os.path.join(root, "data_depth_velodyne", split, drive,
                              "proj_depth", "velodyne_raw", "image_02", f"{frame}.png")
        rgb1   = os.path.join(root, "kitti_raw", split, drive, "proj_depth","image_02", f"{frame}.png")
        rgb2   = os.path.join(root, "kitti_raw", split, drive,"proj_depth", "image_02", f"{frame}.png")
        rgb = _exists(rgb1) or _exists(rgb2)
        sparse = _exists(sparse)
        if not (rgb and sparse and os.path.exists(g)):
            continue
        items.append({"id": f"{drive}/{frame}", "drive": drive, "frame": frame,
                      "rgb": rgb, "sparse": sparse, "gt": g})
    items.sort(key=lambda x: x["id"])
    return items

def build_from_tpl(tpl, root, split, drive, frame):
    return os.path.normpath(tpl.format(root=root, split=split, drive=drive, frame=frame))

def ensure_under_root(path, root):
    path = os.path.abspath(path)
    root = os.path.abspath(root)
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False

def link_file(src, dst, mode="symlink"):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):  # 이미 있으면 건너뜀(링크/파일 모두)
        return
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")

def collect_paths_for_item(it, root, split, pseudo_tpl=None, mono_tpl=None):
    paths = [it["rgb"], it["sparse"], it["gt"]]
    if pseudo_tpl:
        paths.append(build_from_tpl(pseudo_tpl, root, split, it["drive"], it["frame"]))
    if mono_tpl:
        paths.append(build_from_tpl(mono_tpl, root, split, it["drive"], it["frame"]))
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="원본 KITTI-DC root")
    ap.add_argument("--out", required=True, help="샷별 새 루트들이 생성될 상위 폴더")
    ap.add_argument("--shots", type=int, nargs="+", default=[1,10,100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--split", type=str, default="train")

    ap.add_argument("--pseudo-template", type=str, default=None,
                    help="예: {root}/psudo_depth_map/{split}/{drive}/image_02/{frame}.png")
    ap.add_argument("--mono-template", type=str, default=None,
                    help="예: {root}/kitti_raw_da/{split}/{drive}/image_02/{frame}.png")
    ap.add_argument("--check-extra-exists", action="store_true",
                    help="pseudo/mono 실제 존재 프레임만 사용")
    ap.add_argument("--link-mode", choices=["symlink","hardlink","copy"], default="symlink",
                    help="파일 배치 방식(기본: symlink)")
    args = ap.parse_args()

    root_abs = os.path.abspath(args.root)
    items = index_kitti(root_abs, split=args.split)
    assert len(items) > 0, "No frames found. Check directory layout."

    # 필요시 extras 존재성으로 선필터
    if args.check_extra_exists and (args.pseudo_template or args.mono_template):
        before = len(items)
        keep = []
        for it in items:
            ok = True
            if args.pseudo_template:
                p = build_from_tpl(args.pseudo_template, root_abs, args.split, it["drive"], it["frame"])
                ok &= os.path.exists(p)
            if args.mono_template:
                m = build_from_tpl(args.mono_template, root_abs, args.split, it["drive"], it["frame"])
                ok &= os.path.exists(m)
            if ok: keep.append(it)
        items = keep
        assert len(items) > 0, "All frames filtered out by --check-extra-exists."
        print(f"[INFO] Extras check: {before} -> {len(items)}")

    # 상대경로 유지가 가능한지(모든 경로가 root 하위인지) 사전 확인
    def will_be_under_root():
        samples = min(32, len(items))
        idx = np.linspace(0, len(items)-1, samples, dtype=int)
        for i in idx:
            for p in collect_paths_for_item(items[i], root_abs, args.split,
                                            args.pseudo_template, args.mono_template):
                if not ensure_under_root(p, root_abs):
                    return p
        return None
    bad = will_be_under_root()
    if bad:
        raise AssertionError(
            f"상대경로 유지 불가: '{bad}' 가 --root 하위가 아님. "
            "템플릿에서 {root}를 사용해 루트 하위로 맞춰주세요."
        )

    # 샷/시드별로 서브셋 생성 후 링크/복사
    for s in args.seeds:
        rng = np.random.default_rng(s)
        for shot in args.shots:
            assert shot <= len(items), f"shot({shot}) > total({len(items)})"
            subset_idx = rng.choice(len(items), size=shot, replace=False)
            subset = [items[i] for i in sorted(subset_idx)]

            dest_root = os.path.join(args.out, f"{shot}shot", f"S{s}")
            manifest = os.path.join(dest_root, "MANIFEST.txt")
            os.makedirs(dest_root, exist_ok=True)

            # 중복 제거된 파일 집합으로 링크 생성
            file_set = set()
            for it in subset:
                for src in collect_paths_for_item(it, root_abs, args.split,
                                                  args.pseudo_template, args.mono_template):
                    file_set.add(os.path.abspath(src))

            made = 0
            for src in sorted(file_set):
                rel = os.path.relpath(src, root_abs)  # 원본 root 대비 상대경로
                dst = os.path.join(dest_root, rel)    # 새 루트 밑에 동일 상대경로로 배치
                link_file(src, dst, mode=args.link_mode)
                made += 1

            # 매니페스트 기록(참고용)
            with open(manifest, "w") as f:
                f.write(f"root={root_abs}\n")
                f.write(f"split={args.split}\n")
                f.write(f"seed={s}, shot={shot}\n")
                f.write(f"count_items={len(subset)}, count_files={made}\n")
                f.write("frames:\n")
                for it in subset:
                    f.write(f"  {it['id']}\n")

            print(f"[OK] {shot}shot / S{s}: linked {made} files -> {dest_root}")

    print(f"Done. Shots={args.shots}, Seeds={args.seeds}, Mode={args.link_mode}")
    print(f"Switch your dataset ROOT to: {os.path.join(args.out, '<shot>shot', '<Sseed>')}")
if __name__ == "__main__":
    main()
