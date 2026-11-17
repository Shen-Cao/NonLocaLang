"""Measure exact discovery and replay and hash their checked mathematical outputs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from nonlocalgames.core.identity import canonical_json
from nonlocalgames.discovery import canonicalize_xor3, representative_evidence, xor3_game
from nonlocalgames.discovery.provenance import runtime_provenance
from nonlocalgames.discovery.weighted_run import replay_weighted_chsh, run_weighted_chsh
from nonlocalgames.discovery.xor3_corpus import _checked_member


def digest(data):
    return sha256(canonical_json(data).encode()).hexdigest()


def sample_masks():
    groups = {mask: [] for mask in (0, 1, 10)}
    for mask in range(512):
        groups[canonicalize_xor3(xor3_game(mask)).representative].append(mask)
    return tuple(group[(len(group)-1)*i//7] for group in groups.values() for i in range(8))


def measure(masks, representatives):
    with TemporaryDirectory() as temporary:
        path = Path(temporary)
        start = perf_counter()
        saved = run_weighted_chsh(path, steps=20, screen=False)
        generated = perf_counter()
        replayed = replay_weighted_chsh(path)
        finished = perf_counter()
        if saved != replayed:
            raise RuntimeError("Replay disagrees with the generated result")
        science = []
        for k in range(saved["recorded"]):
            data = json.loads((path / "members" / f"{k}.json").read_bytes())
            science.append([data["assessment"], [a["artifact_id"] for a in data["evidence"]]])
    start_xor = perf_counter()
    members = [_checked_member(mask, representatives) for mask in masks]
    elapsed_xor = perf_counter() - start_xor
    return ({"generate_grid": generated-start, "replay_grid": finished-generated,
             "verify_xor3_slice": elapsed_xor},
            {"weighted_chsh": digest(science), "xor3": digest(members)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    masks = sample_masks()
    representatives = {r: representative_evidence(r) for r in (0, 1, 10)}
    _, expected = measure(masks, representatives)
    samples = []
    for _ in range(args.repeats):
        timing, fingerprints = measure(masks, representatives)
        if fingerprints != expected:
            raise RuntimeError("Checked outputs changed across timing samples")
        samples.append(timing)
    print(json.dumps({"schema": "nonlocalgames.exact_replay_benchmark.v1",
                      "provenance": runtime_provenance(None), "checked_output_sha256": expected,
                      "masks": masks, "samples_seconds": samples,
                      "median_seconds": {k: median(s[k] for s in samples) for k in samples[0]}},
                     allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
