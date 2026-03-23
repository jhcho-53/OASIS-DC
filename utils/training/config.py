from config.schema import OasisCfg

def default_iters_for_shots(k):
    if k <= 1:   
        return 600
    if k <= 10:  
        return 1500
    if k <= 100: 
        return 3000
    return 3000

def build_cfg_from_args(args):
    cfg = OasisCfg(
        dmax=args.dmax,
        steps=args.steps,
        kernels=args.kernels,
        geometry=args.geometry,
        use_residual=not args.no_residual,
        use_sparse=not args.no_sparse,
        anchor_learnable=not args.anchor_fixed,
        anchor_mode=args.anchor_mode,
        anchor_alpha=args.anchor_alpha,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
    )
    cfg.use_poisson = (not args.no_poisson)
    cfg.poisson_tol = args.poisson_tol
    cfg.poisson_maxiter = args.poisson_maxiter
    cfg.poisson_init = args.poisson_init
    cfg.poisson_clip_to_max_gt = args.poisson_clip_to_max_gt
    return cfg