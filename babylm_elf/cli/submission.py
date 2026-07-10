from __future__ import annotations

import argparse
from pathlib import Path

from babylm_elf.submission.artifacts import (
    smoke_local_artifact,
    validate_hub_repository,
    validate_revision_tree,
)
from babylm_elf.submission.evaluator import (
    prepare_official_evaluator,
    run_official_evaluation,
)
from babylm_elf.submission.publish import publish_revision_tree


DEFAULT_EVALUATOR_DIR = Path(".cache/babylm-eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, publish, and officially evaluate a BabyLM submission."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Fail-closed artifact or Hub validation.")
    _add_track(check)
    source = check.add_mutually_exclusive_group(required=True)
    source.add_argument("--revisions-dir", type=Path)
    source.add_argument("--repo-id")
    check.add_argument("--smoke", action="store_true")

    publish = commands.add_parser("publish", help="Publish main and all revisions.")
    _add_track(publish)
    publish.add_argument("--revisions-dir", type=Path, required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--private", action="store_true")
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--skip-smoke", action="store_true")

    prepare = commands.add_parser(
        "prepare-eval", help="Clone and pin the unmodified official evaluator."
    )
    prepare.add_argument("--work-dir", type=Path, default=DEFAULT_EVALUATOR_DIR)
    prepare.add_argument("--download-data", action="store_true")

    evaluate = commands.add_parser(
        "evaluate", help="Run full official evaluation for a public Hub model."
    )
    _add_track(evaluate)
    evaluate.add_argument("--repo-id", required=True)
    evaluate.add_argument("--work-dir", type=Path, default=DEFAULT_EVALUATOR_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check":
        if args.revisions_dir:
            tree = validate_revision_tree(args.revisions_dir, args.track)
            if args.smoke:
                smoke_local_artifact(tree.main)
                for _, artifact in tree.revisions:
                    smoke_local_artifact(artifact)
            _print_tree(tree)
        else:
            validate_hub_repository(args.repo_id, args.track)
            print(f"Hub submission is complete: {args.repo_id}")
        return

    if args.command == "publish":
        tree = validate_revision_tree(args.revisions_dir, args.track)
        _print_tree(tree)
        if not args.dry_run:
            publish_revision_tree(
                tree,
                args.repo_id,
                private=args.private,
                smoke=not args.skip_smoke,
            )
        return

    if args.command == "prepare-eval":
        path = prepare_official_evaluator(
            args.work_dir,
            download_data=args.download_data,
        )
        print(f"Official evaluator ready: {path}")
        return

    result = run_official_evaluation(args.repo_id, args.track, args.work_dir)
    print(f"Validated official submission: {result}")


def _add_track(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--track", choices=("strict-small", "strict"), required=True)


def _print_tree(tree) -> None:
    print(f"main <- {tree.main}")
    for revision, artifact in tree.revisions:
        print(f"{revision} <- {artifact}")


if __name__ == "__main__":
    main()
