# make_poisson_nlines_bydate.py
# N-lines(8/32) + Poisson depth completion + date-based calib + crop adjust + viz
# Sampling presets:
#   - firstseq_all : first sequence, ALL frames        <-- NEW
#   - firstseq_one : first sequence, 1 sample
#   - allseq_10    : all sequences, 10 samples (round-robin)
#   - allseq_100   : all sequences, 100 samples (round-robin)
# Saves under:
#   out_root/{8line|32line}/{mode}/{job_tag}/(sparse|poisson|viz)/<rel>.png

import os, re, glob, cv2, numpy as np
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve

# -------------------- I/O --------------------
def load_depth_m(path: str, scale: float) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return (im.astype(np.float32) / float(scale))

def save_depth_u16(path: str, depth_m: np.ndarray, scale: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.clip(depth_m * float(scale), 0, 65535).astype(np.uint16))

# -------------------- Calib by DATE --------------------
def _read_kv_txt(path: str):
    d = {}
    with open(path, 'r') as f:
        for ln in f:
            if ':' not in ln:
                continue
            k, vals = ln.split(':', 1)
            try:
                d[k.strip()] = np.array([float(x) for x in vals.split()], dtype=np.float64)
            except ValueError:
                pass
    return d

def parse_date_from_path(p: str) -> str:
    m = re.search(r'(20\d{2}_\d{2}_\d{2})', p)
    if not m:
        raise ValueError(f"Date (YYYY_MM_DD) not found in path: {p}")
    return m.group(1)

def parse_cam_id_from_path(p: str) -> int:
    m = re.search(r'image_0([23])', p)
    return int(m.group(1)) if m else 2

def load_calibs_by_date(calib_root: str, date_str: str, cam_id: int):
    date_dir = os.path.join(calib_root, date_str)
    cam_txt  = os.path.join(date_dir, 'calib_cam_to_cam.txt')
    velo_txt = os.path.join(date_dir, 'calib_velo_to_cam.txt')
    if not (os.path.exists(cam_txt) and os.path.exists(velo_txt)):
        raise FileNotFoundError(f"Missing calib files under {date_dir}")

    cam_kv  = _read_kv_txt(cam_txt)
    velo_kv = _read_kv_txt(velo_txt)

    P_key = f'P_rect_0{cam_id}'
    if P_key not in cam_kv:
        raise KeyError(f"{P_key} not in {cam_txt}")
    P  = cam_kv[P_key].reshape(3,4)
    R0 = (cam_kv['R_rect_00'] if 'R_rect_00' in cam_kv else cam_kv['R0_rect']).reshape(3,3)

    if 'Tr' in velo_kv:
        Tr = velo_kv['Tr'].reshape(3,4)
    else:
        if 'R' not in velo_kv or 'T' not in velo_kv:
            raise KeyError(f"Tr or (R,T) not in {velo_txt}")
        R = velo_kv['R'].reshape(3,3); T = velo_kv['T'].reshape(3,1)
        Tr = np.hstack([R, T])

    return P.astype(np.float64), R0.astype(np.float64), Tr.astype(np.float64)

# -------------------- Auto-crop adjust --------------------
def adjust_P_for_crop(P: np.ndarray,
                      full_hw: tuple,  # (H_full from EST), (W_full from EST)
                      crop_hw: tuple,  # (H_crop from GT),  (W_crop from GT)
                      crop_w_policy: str = 'center',  # 'center'|'left'|'right'|'none'
                      crop_h_policy: str = 'auto'     # 'auto'|'top'|'center'|'bottom'|'none'
                      ) -> tuple:
    Hf, Wf = full_hw; Hc, Wc = crop_hw
    dx = max(0, Wf - Wc); dy = max(0, Hf - Hc)
    if crop_w_policy == 'none' or dx == 0: left = 0
    elif crop_w_policy == 'left': left = 0
    elif crop_w_policy == 'right': left = dx
    else: left = dx // 2
    if crop_h_policy == 'none' or dy == 0: top = 0
    elif crop_h_policy == 'top': top = 0
    elif crop_h_policy == 'center': top = dy // 2
    elif crop_h_policy == 'bottom': top = dy
    else: top = 0 if (dy % 2 == 1) else (dy // 2)
    P_adj = P.copy()
    P_adj[0,2] -= float(left); P_adj[1,2] -= float(top)
    return P_adj, (left, top)

# -------------------- N-lines (ring) --------------------
def ring_id_from_xyz(x, y, z):
    elev = np.arctan2(z, np.hypot(x, y))
    r = np.round((elev - np.deg2rad(-24.9)) / np.deg2rad(26.9) * 63.0).astype(np.int32)
    return np.clip(r, 0, 63)

def select_mask_for_N_lines(rings, N):
    assert 64 % N == 0, "N must divide 64"
    keep = np.arange(0, 64, 64 // N, dtype=np.int32)
    return np.isin(rings, keep)

def degrade_sparse_to_Nlines_m(sparse_m: np.ndarray,
                               P_adj: np.ndarray,
                               R0_rect: np.ndarray,
                               Tr_velo_to_cam: np.ndarray,
                               N: int) -> np.ndarray:
    mask = sparse_m > 0
    if not np.any(mask): return np.zeros_like(sparse_m, dtype=np.float32)
    v, u = np.where(mask); Z = sparse_m[v, u].astype(np.float64)
    fx, fy = P_adj[0,0], P_adj[1,1]; cx, cy = P_adj[0,2], P_adj[1,2]
    X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy
    T = np.eye(4, dtype=np.float64); T[:3,:4] = Tr_velo_to_cam
    Trect = np.eye(4, dtype=np.float64); Trect[:3,:3] = R0_rect
    T_cam2velo = np.linalg.inv(Trect @ T)
    X_cam  = np.column_stack([X, Y, Z, np.ones_like(Z)])
    X_velo = (T_cam2velo @ X_cam.T).T[:, :3]
    rings = ring_id_from_xyz(X_velo[:,0], X_velo[:,1], X_velo[:,2])
    keep  = select_mask_for_N_lines(rings, N)
    out = np.zeros_like(sparse_m, dtype=np.float32)
    out[v[keep], u[keep]] = sparse_m[v[keep], u[keep]].astype(np.float32)
    return out

# -------------------- Poisson --------------------
def build_Laplacian(H: int, W: int) -> sparse.csr_matrix:
    N = H * W
    main = np.full(N, 4, np.float32)
    off  = np.full(N, -1, np.float32)
    A = sparse.diags([main, off, off, off, off], [0, -1, +1, -W, +W], format='csr')
    for i in range(H):
        L = i * W; R = i * W + (W - 1)
        if L - 1 >= 0: A[L, L-1] = 0; A[L-1, L] = 0
        if R + 1 < N: A[R, R+1] = 0; A[R+1, R] = 0
    return A

def get_A_base(h: int, w: int, cache_dir: str) -> sparse.csr_matrix:
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, f"A_base_{h}x{w}.npz")
    if os.path.exists(fn): return sparse.load_npz(fn)
    A = build_Laplacian(h, w); sparse.save_npz(fn, A); return A

def poisson_complete(sparse_m: np.ndarray, est_m: np.ndarray, A_base: sparse.csr_matrix) -> np.ndarray:
    H, W = est_m.shape
    mask_sparse = (sparse_m > 0)
    mask_border = np.zeros_like(mask_sparse); mask_border[[0,-1],:] = True; mask_border[:,[0,-1]] = True
    mask_known  = (mask_sparse | mask_border).flatten()
    gx = cv2.Sobel(est_m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(est_m, cv2.CV_32F, 0, 1, ksize=3)
    div = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
    b_full = div.flatten()
    d_flat = sparse_m.flatten(); e_flat = est_m.flatten()
    v_known = np.where(mask_sparse.flatten(), d_flat, e_flat)
    idx_k = np.nonzero(mask_known)[0]; idx_u = np.nonzero(~mask_known)[0]
    if idx_u.size == 0:
        return v_known.reshape(H, W).astype(np.float32)
    A_uu = A_base[idx_u][:, idx_u]; A_uk = A_base[idx_u][:, idx_k]
    b_u  = b_full[idx_u] - A_uk.dot(v_known[idx_k])
    x_u  = spsolve(A_uu, b_u).astype(np.float32)
    x = np.empty(H*W, np.float32)
    x[idx_k] = v_known[idx_k].astype(np.float32); x[idx_u] = x_u
    max_gt = float(sparse_m.max()); out = x.reshape(H, W)
    return np.clip(out, 0.0, max_gt) if max_gt > 0 else out

# -------------------- Viz (nearest = purple) --------------------
def build_jet_purple_near_lut() -> np.ndarray:
    base = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)  # (256,1,3) BGR
    lut  = base.copy()
    purple = np.array([128, 0, 128], dtype=np.uint8)  # BGR
    for i in range(224, 256):  # blend top 32 bins toward purple (near)
        a = (i - 224) / 32.0
        lut[i, 0, :] = ((1.0 - a) * purple + a * lut[i, 0, :]).astype(np.uint8)
    return lut

def depth_to_viz_bgr(depth_m: np.ndarray, max_m: float, lut: np.ndarray) -> np.ndarray:
    d = depth_m.copy()
    mask = d > 0
    d = np.clip(d, 0.0, float(max_m))
    val = (1.0 - (d / float(max_m))) * 255.0  # near -> large index
    val = val.astype(np.uint8)
    color = cv2.applyColorMap(val, lut)
    color[~mask] = 0  # zero -> black
    return color

def save_viz_png(path: str, depth_m: np.ndarray, viz_max_m: float, lut: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    color = depth_to_viz_bgr(depth_m, viz_max_m, lut)
    cv2.imwrite(path, color)

# -------------------- Dataset & grouping --------------------
def collect_pairs(root: str, mode: str):
    # GT uses velodyne_raw layout
    ann02 = glob.glob(os.path.join(root, 'data_depth_velodyne', mode, '**/proj_depth/velodyne_raw/image_02/*.png'), recursive=True)
    ann03 = glob.glob(os.path.join(root, 'data_depth_velodyne', mode, '**/proj_depth/velodyne_raw/image_03/*.png'), recursive=True)
    raw02 = glob.glob(os.path.join(root, 'kitti_raw', mode, '**/proj_depth/image_02/*.png'), recursive=True)
    raw03 = glob.glob(os.path.join(root, 'kitti_raw', mode, '**/proj_depth/image_03/*.png'), recursive=True)
    gt_list  = sorted(ann02) + sorted(ann03)
    est_list = sorted(raw02) + sorted(raw03)
    assert len(gt_list) == len(est_list), f"개수 불일치: GT={len(gt_list)} EST={len(est_list)}"
    return gt_list, est_list

def seq_key_from_path(p: str) -> str:
    m = re.search(r'(20\d{2}_\d{2}_\d{2}.*?drive_\d+_sync)', p)
    return m.group(1) if m else os.path.dirname(os.path.dirname(p))

def group_by_sequence(gt_list, est_list):
    pairs = list(zip(gt_list, est_list))
    seqs = {}
    for i, (_, est) in enumerate(pairs):
        k = seq_key_from_path(est)
        seqs.setdefault(k, []).append(i)
    ordered_seq_keys = sorted(seqs.keys())
    for k in ordered_seq_keys:
        seqs[k] = sorted(seqs[k])
    return pairs, ordered_seq_keys, seqs

# -------------------- Sampling presets --------------------
def pick_firstseq_all(ordered_seq_keys, seqs):
    if not ordered_seq_keys: return []
    first_key = ordered_seq_keys[0]
    return list(seqs[first_key])  # ALL frames in first sequence

def pick_firstseq_one(ordered_seq_keys, seqs):
    if not ordered_seq_keys: return []
    first_key = ordered_seq_keys[0]
    return [seqs[first_key][0]]

def pick_allseq_k_roundrobin(ordered_seq_keys, seqs, k):
    sel = []
    max_len = max((len(seqs[k_]) for k_ in ordered_seq_keys), default=0)
    for t in range(max_len):
        for key in ordered_seq_keys:
            if t < len(seqs[key]):
                sel.append(seqs[key][t])
                if len(sel) >= k:
                    return sel
    return sel

def resolve_job_indices(job_tag, ordered_seq_keys, seqs):
    if job_tag == 'firstseq_all':
        return pick_firstseq_all(ordered_seq_keys, seqs)
    if job_tag == 'firstseq_one':
        return pick_firstseq_one(ordered_seq_keys, seqs)
    if job_tag.startswith('allseq_'):
        try:
            k = int(job_tag.split('_', 1)[1])
        except Exception:
            raise ValueError(f"invalid job tag: {job_tag}")
        return pick_allseq_k_roundrobin(ordered_seq_keys, seqs, k)
    raise ValueError(f"unknown job tag: {job_tag}")

# -------------------- Main pipeline --------------------
def run(root: str, out_root: str, calib_root: str, modes, n_list, jobs, gt_scale, est_scale,
        crop_w_policy: str, crop_h_policy: str, viz_max_m: float):
    calib_cache = {}  # (date, cam_id) -> (P, R0, Tr)
    lut = build_jet_purple_near_lut()

    for mode in modes:
        gt_list, est_list = collect_pairs(root, mode)
        pairs, ordered_seq_keys, seqs = group_by_sequence(gt_list, est_list)

        A_cache = {}  # Laplacian cache per (H, W)

        # 공통 처리 함수
        def _process_indices(indices, job_tag, Ns):
            if not indices:
                print(f"[{mode}] job={job_tag}: no samples found.")
                return
            pbar = tqdm(indices, desc=f"{mode} | {job_tag} | {len(indices)} samples")
            for idx in pbar:
                gt_p, est_p = pairs[idx]

                # load depths
                sparse_m = load_depth_m(gt_p,  gt_scale)
                est_m    = load_depth_m(est_p, est_scale)
                Hc, Wc   = sparse_m.shape
                Hf, Wf   = est_m.shape

                if (Hc, Wc) not in A_cache:
                    A_cache[(Hc, Wc)] = get_A_base(Hc, Wc, cache_dir=out_root)
                A_base = A_cache[(Hc, Wc)]

                # date-based calib
                date_str = parse_date_from_path(est_p)
                cam_id   = parse_cam_id_from_path(est_p)
                key = (date_str, cam_id)
                if key not in calib_cache:
                    calib_cache[key] = load_calibs_by_date(calib_root, date_str, cam_id)
                P, R0, Tr = calib_cache[key]

                # crop compensation
                P_adj, _ = adjust_P_for_crop(P, (Hf, Wf), (Hc, Wc), crop_w_policy, crop_h_policy)

                # relative output path kept from data_depth_velodyne/<mode>
                rel = os.path.relpath(gt_p, os.path.join(root, 'data_depth_velodyne', mode))

                for N in Ns:
                    assert 64 % N == 0, f"N must divide 64 (got {N})"
                    # N-lines sparse + Poisson
                    sparse_N = degrade_sparse_to_Nlines_m(sparse_m, P_adj, R0, Tr, N)
                    pseudo_N = poisson_complete(sparse_N, est_m, A_base)

                    # SAVE depths under separate top-level dirs + job tag
                    top = f"{N}line"
                    base_dir = os.path.join(out_root, top, mode, job_tag)
                    save_depth_u16(os.path.join(base_dir, 'sparse',  rel), sparse_N, gt_scale)
                    save_depth_u16(os.path.join(base_dir, 'poisson', rel), pseudo_N, gt_scale)

                    # SAVE viz (near = purple)
                    viz_base = os.path.join(base_dir, 'viz')
                    save_viz_png(os.path.join(viz_base, 'sparse',  rel), sparse_N, viz_max_m, lut)
                    save_viz_png(os.path.join(viz_base, 'poisson', rel), pseudo_N, viz_max_m, lut)

        # ---- 동작 분기 ----
        if mode == 'val':
            # ✅ val은 fullset만 생성: 전체 인덱스 + N=[8,32]
            indices_all = []
            for k in ordered_seq_keys:
                indices_all.extend(seqs[k])
            _process_indices(indices_all, job_tag='fullset', Ns=[8, 32])
        else:
            # train 등은 기존 preset대로
            for job_tag in jobs:
                indices = resolve_job_indices(job_tag, ordered_seq_keys, seqs)
                _process_indices(indices, job_tag, n_list)

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dataset root containing data_depth_velodyne/ and kitti_raw/')
    ap.add_argument('--out_root', required=True, help='output root dir')
    ap.add_argument('--calib_root', required=True, help='calibration root dir organized by date folders')
    ap.add_argument('--modes', nargs='+', default=['train','val'])
    ap.add_argument('--n', nargs='+', type=int, default=[32, 8], help='N-lines list, e.g., --n 32 8')
    ap.add_argument('--jobs', nargs='+', default=['firstseq_all','firstseq_one','allseq_10','allseq_100'],
                    help='sampling presets: firstseq_all, firstseq_one, allseq_10, allseq_100')
    ap.add_argument('--gt_scale', type=float, default=256.0)
    ap.add_argument('--est_scale', type=float, default=256.0)
    ap.add_argument('--crop_w_policy', default='center', choices=['none','left','center','right'])
    ap.add_argument('--crop_h_policy', default='auto',   choices=['none','top','center','bottom','auto'])
    ap.add_argument('--viz_max_m', type=float, default=80.0, help='depth cap (meters) used to normalize viz colors')
    args = ap.parse_args()

    run(args.root, args.out_root, args.calib_root, modes=args.modes, n_list=args.n, jobs=args.jobs,
        gt_scale=args.gt_scale, est_scale=args.est_scale,
        crop_w_policy=args.crop_w_policy, crop_h_policy=args.crop_h_policy,
        viz_max_m=args.viz_max_m)
