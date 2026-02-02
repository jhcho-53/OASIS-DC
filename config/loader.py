from __future__ import annotations
import os, json
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, Tuple, List, Optional

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

from .defaults import DEFAULT_CFG
from .schema import OneShotPaths, TrainCfg, MCPropCfg, LossW

def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """dict deep-merge: override가 base를 재귀적으로 덮어씀"""
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: Optional[str]) -> Dict[str, Any]:
    """defaults.py의 DEFAULT_CFG를 바탕으로 파일을 딥 머지"""
    cfg = deepcopy(DEFAULT_CFG)
    if not path:
        return cfg

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    # 확장자에 따라 JSON/YAML 파싱
    ext = os.path.splitext(path)[-1].lower()
    with open(path, "r") as f:
        if ext in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise RuntimeError("pyyaml이 필요합니다. `pip install pyyaml`")
            user = yaml.safe_load(f)
        else:
            user = json.load(f)

    cfg = _deep_update(cfg, user)
    return cfg

def _filter_kwargs(dc_cls, src: Dict[str, Any]) -> Dict[str, Any]:
    """dataclass가 알고 있는 필드만 추려서 반환 (알 수 없는 키 무시)"""
    if src is None:
        return {}
    valid = {f.name for f in dataclass_fields(dc_cls)}
    return {k: v for k, v in src.items() if k in valid}

def to_dataclasses(cfg: Dict[str, Any]) -> Tuple[OneShotPaths, TrainCfg, MCPropCfg, LossW, List[Dict[str, Any]]]:
    """dict -> dataclass로 변환 (모르는 키는 자동 무시)"""
    # ---- paths ----
    p = cfg.get("paths", {})
    paths_dc = OneShotPaths(
        rgb=p["rgb"],
        sparse=p["sparse"],
        pseudo=p["pseudo"],
        estim=p["estim"],
        gt=p.get("gt", None)
    )

    # ---- train ----
    t = _filter_kwargs(TrainCfg, cfg.get("train", {}))
    train_dc = TrainCfg(**t)

    # ---- model ----
    m = _filter_kwargs(MCPropCfg, cfg.get("model", {}))
    # kernels가 list로 들어오면 tuple로 변환
    if "kernels" in cfg.get("model", {}):
        ks = cfg["model"]["kernels"]
        m["kernels"] = tuple(ks) if isinstance(ks, (list, tuple)) else (3, 5, 7)
    model_dc = MCPropCfg(**m)

    # ---- loss ----
    l = _filter_kwargs(LossW, cfg.get("loss", {}))
    loss_dc = LossW(**l)

    # ---- experiments (옵션) ----
    exps = cfg.get("experiments", [])
    if not isinstance(exps, list):
        exps = []

    return paths_dc, train_dc, model_dc, loss_dc, exps