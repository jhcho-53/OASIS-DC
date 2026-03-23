#!/usr/bin/env python3
import os, re, glob, argparse, shutil
from typing import List, Tuple
from tqdm import tqdm

SEQ_RE = re.compile(r'(20\d{2}_\d{2}_\d{2}[^/]*?drive_\d+_sync)')
CAM_RE = re.compile(r'image_0([0-9])')

def list_pngs(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, '**', '*.png'), recursive=True))

def parse_seq_key(path: str) -> str:
    m = SEQ_RE.search(path)
    return m.group(1) if m else "unknown_seq"

def parse_cam_id(path: str) -> str:
    m = CAM_RE.search(path)
    return m.group(1) if m else ""

def parse_frame_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    # 파일명 내 마지막 숫자 시퀀스를 프레임ID로 간주
    m = re.findall(r'\d+', stem)
    if m:
        return f"{int(m[-1]):010d}"
    return stem  # 숫자가 없으면 그대로

def compose_name(idx: int, seq: str, frame: str,
                 cam: str, tag: str,
                 cam_mode: str) -> str:
    """
    cam_mode: 'omit'|'suffix'|'prefix'
    tag: 파일명 마지막에 '_{tag}'로 붙임(예: 8l, 32l)
    """
    parts = [f"{idx:06d}", seq, frame]
    if cam and cam_mode == 'prefix':
        parts.insert(1, f"0{cam}")
    elif cam and cam_mode == 'suffix':
        parts[-1] = f"{parts[-1]}_0{cam}"
    if tag:
        parts[-1] = f"{parts[-1]}_{tag}"
    return "_".join(parts) + ".png"

def safe_link_copy_move(src: str, dst: str, mode: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return False  # 충돌은 호출부에서 처리
    if mode == 'link':
        # 절대경로 심볼릭 링크
        os.symlink(os.path.abspath(src), dst)
    elif mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'move':
        shutil.move(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return True

def auto_start_idx(dst: str) -> int:
    # 대상 디렉토리에 이미 있는 png 개수 기반으로 시작 인덱스 추정
    if not os.path.isdir(dst):
        return 0
    return len(glob.glob(os.path.join(dst, '*.png')))

def main():
    ap = argparse.ArgumentParser("Flatten KITTI sparse (8/32-lines) into a single folder with renamed files")
    ap.add_argument("--src", nargs="+", required=True,
                    help="소스 폴더(여러 개 가능). 예: /.../data/8line/train/allseq_10/sparse")
    ap.add_argument("--dst", required=True,
                    help="타깃 폴더. 예: /.../data/kitti_k10_dataset/train/sparse")
    ap.add_argument("--mode", default="link", choices=["link","copy","move"],
                    help="파일 처리 방식 (기본: link)")
    ap.add_argument("--start-idx", type=int, default=-1,
                    help="글로벌 카운터 시작값(기본: 자동)")
    ap.add_argument("--cam-mode", default="omit", choices=["omit","suffix","prefix"],
                    help="카메라ID(image_0X)를 파일명에 포함 여부/위치 (기본 omit)")
    ap.add_argument("--tag", default="", help="선택: 파일명에 붙일 태그 (예: 8l, 32l)")
    ap.add_argument("--dry-run", action="store_true", help="실제 쓰기 없이 미리보기")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    idx = args.start_idx if args.start_idx >= 0 else auto_start_idx(args.dst)

    used_names = set(x for x in map(os.path.basename, glob.glob(os.path.join(args.dst, '*.png'))))

    total = 0
    for src_root in args.src:
        files = list_pngs(src_root)
        pbar = tqdm(files, desc=f"Scanning {src_root}", unit="img")
        for src in pbar:
            seq   = parse_seq_key(src)
            frame = parse_frame_id(src)
            cam   = parse_cam_id(src)

            name  = compose_name(idx, seq, frame, cam, args.tag, args.cam_mode)
            dst   = os.path.join(args.dst, name)

            # 충돌 처리: 동일 이름 존재 시 cam/dup suffix 추가
            dup_try = 0
            cur_name, cur_dst = name, dst
            while os.path.basename(cur_dst) in used_names or os.path.exists(cur_dst):
                dup_try += 1
                # cam 정보가 없고 충돌이면 cam suffix를 먼저 고려
                extra = ""
                if args.cam_mode == "omit" and cam:
                    extra = f"_0{cam}"
                cur_name = os.path.splitext(name)[0] + f"{extra}_dup{dup_try}.png"
                cur_dst  = os.path.join(args.dst, cur_name)
            name, dst = cur_name, cur_dst

            if args.dry_run:
                pbar.set_postfix_str(f"{os.path.relpath(src, src_root)} -> {name}")
            else:
                ok = safe_link_copy_move(src, dst, args.mode)
                if ok:
                    used_names.add(name)
                    idx += 1
                    total += 1
                pbar.set_postfix_str(name)

    print(f"[DONE] wrote {total} files to: {args.dst}")

if __name__ == "__main__":
    main()
