#!/usr/bin/env python3
import argparse, pathlib, re, random, json, sys
from collections import defaultdict

DRIVE_RE = re.compile(r'(2011_\d{2}_\d{2}_drive_\d{4}_sync)')
CAM_RE   = re.compile(r'(image_0[23])')

def iter_gt_files(gt_root, cams):
    root = pathlib.Path(gt_root)
    iters = []
    if cams in ('image_02', 'image_03'):
        iters = [root.glob(f'**/proj_depth/groundtruth/{cams}/*.png')]
    elif cams == 'both':
        iters = [
            root.glob('**/proj_depth/groundtruth/image_02/*.png'),
            root.glob('**/proj_depth/groundtruth/image_03/*.png')
        ]
    files = sorted(set(p.resolve() for g in iters for p in g))
    if not files:
        raise SystemExit(f'No GT pngs found under {gt_root}')
    return files

def extract_keys(gt_path: pathlib.Path):
    s = str(gt_path)
    m1, m2 = DRIVE_RE.search(s), CAM_RE.search(s)
    if not (m1 and m2):
        raise ValueError(f'Cannot parse drive/cam from: {s}')
    return m1.group(1), m2.group(1), gt_path.name  # drive, camera, frame(000000.png)

def build_path(root, pattern, drive, camera, frame):
    return pathlib.Path(root) / pattern.format(
        drive=drive, camera=camera, frame=frame, frame_noext=pathlib.Path(frame).stem
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-root',     required=True, help='.../data_depth_annotated/val')
    ap.add_argument('--rgb-root',    required=True, help='.../data_rgb/val (KITTI raw layout)')
    ap.add_argument('--sparse-root', required=True, help='.../data_depth_velodyne/val')
    ap.add_argument('--pseudo-root', required=True, help='.../pseudo_depth/val (your dir)')
    ap.add_argument('--outdir',      required=True)
    ap.add_argument('--cams', default='image_02', choices=['image_02','image_03','both'])
    ap.add_argument('--report-ratio', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--relative-to', default='', help='make paths relative to this dir')
    ap.add_argument('--skip-missing', action='store_true',
                    help='skip samples missing any of the 4 files (default: error)')
    # Path patterns relative to each root (edit if your layout differs)
    ap.add_argument('--rgb-pattern',    default='{drive}/proj_depth/{camera}/{frame}')
    ap.add_argument('--sparse-pattern', default='{drive}/proj_depth/velodyne_raw/{camera}/{frame}')
    ap.add_argument('--pseudo-pattern', default='{drive}/proj_depth/velodyne_raw/{camera}/{frame}')
    args = ap.parse_args()

    gt_files = iter_gt_files(args.gt_root, args.cams)
    base = pathlib.Path(args.relative_to).resolve() if args.relative_to else None

    rows, missing = [], []
    for gt in gt_files:
        drive, cam, frame = extract_keys(gt)
        rgb = build_path(args.rgb_root,    args.rgb_pattern,    drive, cam, frame)
        sp  = build_path(args.sparse_root, args.sparse_pattern, drive, cam, frame)
        ps  = build_path(args.pseudo_root, args.pseudo_pattern, drive, cam, frame)
        if rgb.exists() and sp.exists() and ps.exists() and gt.exists():
            rows.append((drive, rgb.resolve(), sp.resolve(), gt.resolve(), ps.resolve()))
        else:
            missing.append({
                'gt': str(gt),
                'rgb': str(rgb), 'rgb_exists': rgb.exists(),
                'sparse': str(sp), 'sparse_exists': sp.exists(),
                'pseudo': str(ps), 'pseudo_exists': ps.exists(),
            })

    if missing and not args.skip_missing:
        print(f'[ERROR] Missing {len(missing)} samples. Example:\n' +
              json.dumps(missing[:5], indent=2), file=sys.stderr)
        sys.exit(2)

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    def pstr(p): return str(p.relative_to(base) if base else p)

    # val_all_4col.txt  (RGB \t Sparse \t GT \t Pseudo)
    with open(outdir/'val_all_4col.txt','w') as f:
        for _, rgb, sp, gt, ps in rows:
            f.write('\t'.join([pstr(rgb), pstr(sp), pstr(gt), pstr(ps)]) + '\n')

    # group by drive → 80/20 split (greedy, large drives first)
    groups = defaultdict(list)
    for drive, rgb, sp, gt, ps in rows:
        groups[drive].append((rgb, sp, gt, ps))
    total = sum(len(v) for v in groups.values())
    target = int(round(total * args.report_ratio))
    items = list(groups.items())
    rnd = random.Random(args.seed); rnd.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    report, tune, acc, report_drives, tune_drives = [], [], 0, [], []
    for d, lst in items:
        if acc < target:
            report.extend(lst); acc += len(lst); report_drives.append(d)
        else:
            tune.extend(lst); tune_drives.append(d)

    def write_4col(path, data):
        with open(path, 'w') as f:
            for rgb, sp, gt, ps in sorted(data):
                f.write('\t'.join([pstr(rgb), pstr(sp), pstr(gt), pstr(ps)]) + '\n')

    write_4col(outdir/'val_report_4col.txt', report)
    write_4col(outdir/'val_tune_4col.txt',   tune)

    meta = {
        'counts': {
            'gt_found': len(gt_files),
            'rows_kept': len(rows),
            'missing': len(missing),
            'report': len(report),
            'tune': len(tune),
        },
        'report_ratio_effective': len(report)/max(1, (len(report)+len(tune))),
        'seed': args.seed,
        'cams': args.cams,
        'roots': {
            'gt_root': str(pathlib.Path(args.gt_root).resolve()),
            'rgb_root': str(pathlib.Path(args.rgb_root).resolve()),
            'sparse_root': str(pathlib.Path(args.sparse_root).resolve()),
            'pseudo_root': str(pathlib.Path(args.pseudo_root).resolve()),
        },
        'patterns': {
            'rgb': args.rgb_pattern, 'sparse': args.sparse_pattern, 'pseudo': args.pseudo_pattern
        },
        'report_drives': report_drives, 'tune_drives': tune_drives,
        'relative_to': str(base) if base else None,
        'skipped_examples': missing[:10],
    }
    (outdir/'val_split_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
