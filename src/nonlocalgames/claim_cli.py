"""Replay a portable value or gap definition against a verifier and raw artifacts."""
import argparse
import json
from pathlib import Path

from .claims import (GapClaim, VerificationBudget, assess_gap, assess_value, assessment_to_dict,
                     claim_from_dict, evidence_from_dict)
from .core.json_file import read_json
from .core.serialization import from_dict as game_from_dict


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nonlocalgames claim", description=__doc__)
    parser.add_argument("game", type=Path, help="finite verifier JSON")
    parser.add_argument("definition", type=Path, help="versioned claim definition bound to the verifier")
    parser.add_argument("evidence", type=Path, help="JSON array of raw evidence envelopes")
    parser.add_argument("--budget", type=Path, help="JSON object of verification budget overrides")
    parser.add_argument("--max-file-bytes", type=int, default=8_000_000)
    parser.add_argument("--max-game-entries", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    try:
        budget = VerificationBudget() if args.budget is None else VerificationBudget(
            **read_json(args.budget, max_file_bytes=16_384))
        limits = {"max_scalar_bits": budget.max_wire_scalar_bits,
                  "max_scalar_chars": budget.max_wire_scalar_chars}
        game = game_from_dict(read_json(args.game, max_file_bytes=args.max_file_bytes),
                              max_entries=args.max_game_entries, **limits)
        claim = claim_from_dict(game, read_json(args.definition, max_file_bytes=args.max_file_bytes),
                                max_bytes=args.max_file_bytes, **limits)
        raw = read_json(args.evidence, max_file_bytes=args.max_file_bytes)
        if not isinstance(raw, list) or len(raw) > budget.max_artifacts:
            raise ValueError("Provide an evidence array within max_artifacts")
        artifacts = [evidence_from_dict(item, max_bytes=budget.max_artifact_bytes) for item in raw]
        assess = assess_gap if isinstance(claim, GapClaim) else assess_value
        receipt = assessment_to_dict(assess(game, claim, artifacts, budget=budget))
    except (TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2, allow_nan=False))
