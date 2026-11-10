"""Shared validation for the independent depth-loop and NextLat experiments."""
import math


LOOP_DEFAULTS = {
    "loop_iterations": 1,
    "loop_start_layer": None,
    "loop_end_layer": None,
    "loop_input_injection": True,
    # Learned-loop extensions. Every option is the identity at initialization:
    # a model with any of them starts numerically equal to the plain loop.
    "loop_dense_mixing": False,        # earlier-iteration latents into later restarts and the coda
    "loop_iteration_embedding": False,  # learned per-restart offset
    "loop_gate": "none",                # none | static | token gate on each looped block update
    "loop_gate_init": 1.0,              # initial gate value in (0, 2); 1 keeps the update unchanged
    "loop_ffn": "all",                  # all | first: later iterations run attention only
    "loop_iteration_norms": False,      # per-iteration gains on the shared blocks' norms
    "loop_streams": 0,                  # 0 or >=2 residual streams written once per iteration
    "loop_stream_gate": "dynamic",      # static | dynamic diagonal keep-gate per stream
    "loop_exit_weight": 0.0,            # deep supervision: early-exit CE weight after iterations 1..K-1
}
LOOP_ENUMS = {
    "loop_gate": ("none", "static", "token"),
    "loop_ffn": ("all", "first"),
    "loop_stream_gate": ("static", "dynamic"),
}
LOOP_EXTENSION_DEFAULTS = {key: LOOP_DEFAULTS[key] for key in (
    "loop_dense_mixing", "loop_iteration_embedding", "loop_gate", "loop_ffn",
    "loop_iteration_norms", "loop_streams", "loop_exit_weight")}
NEXT_LATENT_DEFAULTS = {
    "next_latent_prediction": False,
    "next_latent_horizon": 1,
    "next_latent_hidden_dim": None,
    "next_latent_regression_weight": 1.0,
    "next_latent_kl_weight": 1.0,
}
RESEARCH_DEFAULTS = {**LOOP_DEFAULTS, **NEXT_LATENT_DEFAULTS}


def validate_research_options(options, *, n_layers=None, require_complete=False):
    """Allow omitted/None launcher values to inherit a checkpoint's settings."""
    for key in ("loop_iterations", "loop_start_layer", "loop_end_layer", "next_latent_hidden_dim", "next_latent_horizon"):
        value = options.get(key)
        minimum = 0 if key in ("loop_start_layer", "loop_end_layer") else 1
        if require_complete and key in ("loop_iterations", "next_latent_horizon") and value is None:
            raise ValueError(f"{key} must be an integer >= 1.")
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    for key in ("loop_input_injection", "next_latent_prediction", "loop_dense_mixing",
                "loop_iteration_embedding", "loop_iteration_norms"):
        value = options.get(key)
        if (require_complete or value is not None) and not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean.")
    for key in ("next_latent_regression_weight", "next_latent_kl_weight"):
        value = options.get(key)
        if (require_complete or value is not None) and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be finite and >= 0.")
    for key, allowed in LOOP_ENUMS.items():
        value = options.get(key)
        if (require_complete or value is not None) and value not in allowed:
            raise ValueError(f"{key} must be one of {', '.join(allowed)}.")
    value = options.get("loop_gate_init")
    if (require_complete or value is not None) and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value < 2):
        raise ValueError("loop_gate_init must be finite and strictly between 0 and 2.")
    value = options.get("loop_exit_weight")
    if (require_complete or value is not None) and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("loop_exit_weight must be finite and >= 0.")
    value = options.get("loop_streams")
    if (require_complete or value is not None) and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0 or value == 1):
        raise ValueError("loop_streams must be 0 or an integer >= 2.")
    if options.get("loop_iterations") == 1:
        changed = [key for key, default in LOOP_EXTENSION_DEFAULTS.items()
                   if options.get(key) is not None and options.get(key) != default]
        if changed:
            raise ValueError("loop_iterations must exceed 1 to use " + ", ".join(changed) + ".")
    start, end = options.get("loop_start_layer"), options.get("loop_end_layer")
    if start is not None and end is not None:
        if start >= end or (n_layers is not None and end > n_layers):
            raise ValueError("loop_start_layer and loop_end_layer must select a nonempty layer span within n_layers.")


def research_parameter_shapes(config):
    """Expected new checkpoint tensors, including when converting objectives."""
    shapes = {}
    if config.loop_iterations > 1 and config.loop_input_injection:
        shapes["loop_input_scale"] = ()
    shapes.update(loop_extension_shapes(config))
    if config.next_latent_prediction:
        width = config.d_model
        hidden = config.next_latent_hidden_dim or 2 * width
        shapes.update({
            "next_latent_dynamics.input_norm.weight": (2 * width,),
            "next_latent_dynamics.mlp.0.weight": (hidden, 2 * width),
            "next_latent_dynamics.mlp.2.weight": (hidden, hidden),
            "next_latent_dynamics.mlp.4.weight": (width, hidden),
        })
    return shapes


def loop_dense_pairs(iterations):
    """(restart iteration k, earlier iteration j) pairs beyond the prelude injection."""
    return tuple((k, j) for k in range(3, iterations + 1) for j in range(1, k - 1))


def loop_extension_shapes(config):
    iterations = config.loop_iterations
    if iterations <= 1:
        return {}
    width = config.d_model
    span = config.loop_end_layer - config.loop_start_layer
    shapes = {}
    if config.loop_dense_mixing:
        pairs = loop_dense_pairs(iterations)
        if pairs:
            shapes["loop_dense_scales"] = (len(pairs), width)
        shapes["loop_coda_scales"] = (iterations, width)
    if config.loop_iteration_embedding:
        shapes["loop_iteration_embeddings"] = (iterations - 1, width)
    if config.loop_gate != "none":
        shapes["loop_gate_bias"] = (iterations * span,)
        if config.loop_gate == "token":
            shapes["loop_gate_weight"] = (iterations * span, width)
    if config.loop_iteration_norms:
        shapes["loop_norm_gains"] = ((iterations - 1) * span, 2, width)
    if config.loop_streams:
        streams = config.loop_streams
        shapes["loop_stream_read"] = (iterations + 1, streams)
        shapes["loop_stream_write"] = (iterations, streams)
        shapes["loop_stream_gate_bias"] = (iterations, streams)
        if config.loop_stream_gate == "dynamic":
            shapes["loop_stream_gate_weight"] = (iterations, streams, width)
    return shapes
