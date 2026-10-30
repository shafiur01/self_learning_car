"""Mathematical checks for a proposed admission rule; NOT a gesture experiment.

All teacher logits and anchor responses below are hand-constructed synthetic
numbers. No recordings, hidden labels, saved models or real accuracy are read.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .run import PROJECT, sha256, write_json

BLOCKS = {
    "pose": (*range(0, 63), *range(134, 143)),
    "articulation": tuple(range(67, 130)),
    "transport": (*range(63, 66), *range(130, 133)),
    "context": (66, 133, 143, 144, 145),
}


def response(logits, class_index):
    """Four views: all, omit pose, omit articulation, omit transport."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (4, 8) or not np.isfinite(logits).all():
        raise ValueError("Need four finite eight-class logit rows.")
    other = np.delete(logits, class_index, axis=1)
    maximum = other.max(axis=1)
    rest = maximum + np.log(np.exp(other - maximum[:, None]).sum(axis=1))
    odds = logits[:, class_index] - rest
    return odds[0] - odds[1:]


def compatibility(candidate, anchors, scale):
    """Provisional heuristic: minimum three trusted disjoint same-class anchors."""
    anchors = np.asarray(anchors, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 3 or len(anchors) < 3:
        return {"supported": False, "accepted": False, "weight": 0.0}
    scale = np.maximum(np.asarray(scale, dtype=np.float64), .05)
    pairwise = np.sum(((anchors[:, None] - anchors[None, :]) / scale) ** 2, axis=2)
    np.fill_diagonal(pairwise, np.inf)
    radius = max(float(np.median(pairwise.min(axis=1))), .05 ** 2)
    distance = float(np.min(np.sum(((anchors - np.asarray(candidate)) / scale) ** 2, axis=1)))
    accepted = distance <= radius
    return {"supported": True, "accepted": accepted, "weight": float(np.exp(-distance / radius)) if accepted else 0.0,
            "distance_squared": distance, "radius_squared": radius}


def synthetic_logits(confidence, signature, class_index=0):
    """Generate synthetic logits realizing a requested response; no fitted model."""
    odds = np.log(confidence) - np.log1p(-confidence)
    logits = np.zeros((4, 8))
    logits[:, class_index] = np.r_[odds, odds - np.asarray(signature)] + np.log(7)
    return logits


def check():
    checks = []
    indices = [index for block in BLOCKS.values() for index in block]
    assert sorted(indices) == list(range(146)) and len(set(indices)) == 146
    checks.append("All146 existing channels assigned exactly once; no omitted or duplicated numerical channel.")

    # A proper subset posterior can be ambiguous; agreement with full input is not forced.
    p = np.linspace(.01, .99, 99)
    ambiguous_loss = -.5 * np.log(p) - .5 * np.log1p(-p)
    assert np.isclose(p[ambiguous_loss.argmin()], .5)
    checks.append("Identical partial observations from two balanced labels minimize CE at50/50, not forced full-label confidence.")

    offsets = np.array([[-.08, -.15, -.04], [-.04, -.06, .02], [0, 0, 0], [.04, .06, -.02], [.08, .15, .04]])
    forward = np.array([.2, 3.0, .1]) + offsets
    up = np.array([3.0, .1, .1]) + offsets
    bank = np.r_[forward, up]
    scale = np.maximum(1.4826 * np.median(np.abs(bank - np.median(bank, axis=0)), axis=0), .05)
    correct_motion = compatibility(response(synthetic_logits(.99, [.205, 3.005, .105]), 0), forward, scale)
    wrong_basis = compatibility(response(synthetic_logits(.99, [3.0, .1, .1]), 0), forward, scale)
    assert correct_motion["accepted"] and not wrong_basis["accepted"]
    checks.append("Two equal99% synthetic confidences can receive different admission decisions based on response compatibility.")

    static_up = compatibility(response(synthetic_logits(.97, [3.005, .105, .105]), 0), up, scale)
    assert static_up["accepted"]
    checks.append("Held-UP-like synthetic profile accepted despite weak motion contribution; no universal motion requirement.")

    negative = response(synthetic_logits(.99, [.2, -.3, .1]), 0)
    np.testing.assert_allclose(negative, [.2, -.3, .1], atol=1e-12)
    checks.append("Signed responses preserved: removing a misleading group may increase class support.")
    unsupported = compatibility([0, 0, 0], forward[:2], scale)
    assert not unsupported["supported"] and not unsupported["accepted"]
    checks.append("Insufficient anchors produce unsupported admission, not a fabricated command label.")

    # A deliberately wrong prediction with the same profile is indistinguishable.
    shared_error = compatibility(response(synthetic_logits(.99, [.205, 3.005, .105]), 0), forward, scale)
    assert shared_error["accepted"]
    checks.append("Counterexample retained: a wrong prediction with a trusted-looking profile is still accepted.")

    error = np.array([0., 1., 0., 1.])
    weight = np.array([1., .2, .8, .1])
    base = error.mean()
    weighted = np.mean(weight * error) / weight.mean()
    covariance = np.mean((weight - weight.mean()) * (error - base))
    np.testing.assert_allclose(weighted - base, covariance / weight.mean(), atol=1e-14)
    checks.append("Weighted-error identity verified; improvement requires negative error/weight covariance and is not guaranteed.")
    return {"scope": "Synthetic mathematical sanity checks only; no learned gesture model, actual accuracy, novelty proof or label savings.",
            "checks_passed": len(checks), "checks": checks,
            "constructed_examples": {"motion_supported": correct_motion, "same_confidence_wrong_basis": wrong_basis,
                                     "held_shape_supported": static_up, "shared_error_not_detected": shared_error},
            "risk_identity": {"synthetic_base_error": base, "synthetic_weighted_error": weighted,
                              "covariance": covariance, "note": "Hand-constructed illustration, not measured gesture results."}}


def main():
    result = check()
    output = PROJECT / "research/experiments/runs/evidence_profile_design" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    (output / source.name).write_bytes(source.read_bytes())
    write_json(output / "mathematical_checks.json", result)
    write_json(output / "provenance.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": sha256(source), "real_data_accessed": False, "real_model_training": False})
    write_json(output.parent / "latest.json", {"folder": str(output)})
    print(json.dumps({"checks_passed": result["checks_passed"], "output": str(output), "scope": result["scope"]}, indent=2))


if __name__ == "__main__":
    main()
