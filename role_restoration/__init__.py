"""Role restoration: a residual-stream defense against prompt injection, and the geometry behind it.

Submodules (import what you need; `plots` and `legacy` do not require torch):
    base          model loading, harmony scaffold, attacks, activation capture, role means, probe   (needs torch)
    geometry      the role-leakage experiment                                                       (needs torch)
    intervention  style ablation, role restoration, controls, benign utility, mediation, figure-1 data (needs torch)
    plots         every figure and table, from saved results
    legacy        loaders for the original Colab / RunPod artefacts
"""
__all__ = ["base", "geometry", "intervention", "plots", "legacy"]
