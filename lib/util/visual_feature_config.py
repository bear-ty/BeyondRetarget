VISUAL_FEATURE_PRESETS = {
    "hmr2_1024": {
        "filename": "vit_features.pt",
        "input_dim": 1024,
        "model_dim": 1024,
    },
}


def resolve_visual_feature_cfg(cfg=None):
    cfg = dict(cfg or {})
    feature_type = str(cfg.get("type", "hmr2_1024"))
    # Older configurations may qualify the encoder name with a pipeline prefix.
    if feature_type.endswith("_hmr2_1024"):
        feature_type = "hmr2_1024"
    if feature_type not in VISUAL_FEATURE_PRESETS:
        valid = ", ".join(sorted(VISUAL_FEATURE_PRESETS))
        raise ValueError(f"Unsupported visual_feature.type={feature_type!r}; expected one of: {valid}")

    resolved = dict(VISUAL_FEATURE_PRESETS[feature_type])
    resolved.update(cfg)
    resolved["type"] = feature_type
    resolved["input_dim"] = int(resolved["input_dim"])
    resolved["model_dim"] = int(resolved["model_dim"])

    preset = VISUAL_FEATURE_PRESETS[feature_type]
    if resolved["input_dim"] != int(preset["input_dim"]):
        raise ValueError(f"{feature_type} expects visual_feature.input_dim={preset['input_dim']}")
    return resolved
