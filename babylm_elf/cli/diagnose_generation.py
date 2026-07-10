from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from babylm_elf.data.token_stream import open_token_stream


DEFAULT_STEPS = (32, 64, 128)
DEFAULT_TIMES = (0.0, 0.25, 0.5, 0.75, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run optional open-ended generation and denoising diagnostics for an "
            "exported BabyLM-ELF HF model. BabyLM official scoring does not use this."
        )
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--methods", nargs="+", default=["sde", "ode"])
    parser.add_argument("--denoise-times", type=float, nargs="+", default=list(DEFAULT_TIMES))
    parser.add_argument("--tokenized-path", type=Path, default=None)
    parser.add_argument("--denoise-batch-size", type=int, default=8)
    parser.add_argument("--denoise-batches", type=int, default=4)
    parser.add_argument("--max-denoise-sequences", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_diagnostics(
        args.model_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        steps=tuple(args.steps),
        methods=tuple(args.methods),
        denoise_times=tuple(args.denoise_times),
        tokenized_path=args.tokenized_path,
        denoise_batch_size=args.denoise_batch_size,
        denoise_batches=args.denoise_batches,
        max_denoise_sequences=args.max_denoise_sequences,
    )
    print(f"Wrote diagnostics to {output_dir}")


def run_diagnostics(
    model_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    device: str = "auto",
    seed: int = 42,
    num_samples: int = 32,
    sequence_length: int | None = None,
    steps: tuple[int, ...] = DEFAULT_STEPS,
    methods: tuple[str, ...] = ("sde", "ode"),
    denoise_times: tuple[float, ...] = DEFAULT_TIMES,
    tokenized_path: str | Path | None = None,
    denoise_batch_size: int = 8,
    denoise_batches: int = 4,
    max_denoise_sequences: int | None = None,
) -> Path:
    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    run_name = run_name or _run_name(config, model_dir)
    output_root = Path(output_dir) if output_dir is not None else Path("diagnostics")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
    ).to(target_device)
    model.eval()

    sample_files = _write_samples(
        model,
        tokenizer,
        run_dir,
        seed=seed,
        num_samples=num_samples,
        sequence_length=sequence_length or int(getattr(config, "max_position_embeddings", 512)),
        steps=steps,
        methods=methods,
    )
    report = _denoise_report(
        model,
        config,
        model_dir,
        tokenized_path=tokenized_path,
        batch_size=denoise_batch_size,
        batches=denoise_batches,
        max_sequences=max_denoise_sequences,
        times=denoise_times,
        seed=seed,
        device=target_device,
    )
    report_path = run_dir / "denoise_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary(run_dir, run_name, sample_files, report)
    return run_dir


@torch.no_grad()
def _write_samples(
    model,
    tokenizer,
    run_dir: Path,
    *,
    seed: int,
    num_samples: int,
    sequence_length: int,
    steps: tuple[int, ...],
    methods: tuple[str, ...],
) -> list[Path]:
    sample_files: list[Path] = []
    for method in methods:
        if method not in {"sde", "ode"}:
            raise ValueError(f"Unsupported sampling method: {method}")
        for num_steps in steps:
            torch.manual_seed(seed)
            diagnostic_generate = getattr(
                model,
                "diagnostic_generate",
                model.generate,
            )
            ids = diagnostic_generate(
                batch_size=num_samples,
                sequence_length=sequence_length,
                num_steps=int(num_steps),
                sampling_method=method,
            )
            lines = [
                tokenizer.decode(row.tolist(), skip_special_tokens=True).strip()
                for row in ids.cpu()
            ]
            path = run_dir / f"samples_{method}_{num_steps}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            sample_files.append(path)
    return sample_files


@torch.no_grad()
def _denoise_report(
    model,
    config,
    model_dir: Path,
    *,
    tokenized_path: str | Path | None,
    batch_size: int,
    batches: int,
    max_sequences: int | None,
    times: tuple[float, ...],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    sequences = _load_denoise_sequences(
        config,
        model_dir,
        tokenized_path,
        batch_size=batch_size,
        batches=batches,
        max_sequences=max_sequences,
    )
    if sequences is None:
        return {
            "status": "skipped",
            "reason": "No tokenized-path was provided or found in training metadata.",
        }

    torch.manual_seed(seed)
    input_ids = sequences.to(device)
    attention_mask = input_ids.ne(int(getattr(config, "pad_token_id", 3))).long()
    valid_mask = attention_mask.bool()
    maskable = valid_mask & input_ids.ge(16)
    clean = model.babylm_elf.embed_tokens(input_ids, attention_mask=attention_mask)
    noise_scale = float(config.diffusion_config.get("denoiser_noise_scale", 2.0))
    t_eps = float(config.diffusion_config.get("t_eps", 0.05))
    rows: list[dict[str, float]] = []
    for t_value in times:
        t = torch.full((input_ids.size(0),), float(t_value), device=device, dtype=clean.dtype)
        noise = torch.randn_like(clean)
        z_t = t.view(-1, 1, 1) * clean + (1.0 - t.view(-1, 1, 1)) * noise * noise_scale
        prediction, _ = model.babylm_elf(
            torch.cat((z_t, torch.zeros_like(z_t)), dim=-1),
            t,
            attention_mask=attention_mask,
            self_cond_cfg_scale=torch.ones_like(t),
            decoder_step_active=torch.zeros_like(t),
        )
        mse = (prediction.float() - clean.float()).square().mean(dim=-1)
        hidden = model._decoder_hidden(
            prediction,
            attention_mask,
            torch.ones_like(t),
        )
        logits = model._decode_hidden(hidden)
        token_ce = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            input_ids.reshape(-1),
            reduction="none",
        ).view_as(input_ids)
        predicted_ids = logits.argmax(dim=-1)
        rows.append(
            {
                "t": float(t_value),
                "mse": _masked_mean(mse, valid_mask),
                "token_accuracy": _masked_mean(
                    predicted_ids.eq(input_ids).float(),
                    valid_mask,
                ),
                "maskable_token_ce": _masked_mean(token_ce, maskable),
                "maskable_token_accuracy": _masked_mean(
                    predicted_ids.eq(input_ids).float(),
                    maskable,
                ),
            }
        )
    return {
        "status": "ok",
        "num_sequences": int(input_ids.size(0)),
        "sequence_length": int(input_ids.size(1)),
        "time_points": rows,
    }


def _load_denoise_sequences(
    config,
    model_dir: Path,
    tokenized_path: str | Path | None,
    *,
    batch_size: int,
    batches: int,
    max_sequences: int | None,
) -> torch.Tensor | None:
    path = Path(tokenized_path) if tokenized_path is not None else _metadata_train_path(config, model_dir)
    if path is None or not path.exists():
        return None
    stream = open_token_stream(path)
    seq_length = int(getattr(config, "max_position_embeddings", 512))
    pad_id = int(getattr(config, "pad_token_id", 3))
    needed = max_sequences or max(1, batch_size * batches)
    rows: list[torch.Tensor] = []
    for index in range(min(needed, (stream.numel() + seq_length - 1) // seq_length)):
        values = stream[index * seq_length : (index + 1) * seq_length].long()
        if values.numel() == seq_length:
            rows.append(values)
            continue
        row = torch.full((seq_length,), pad_id, dtype=torch.long)
        row[: values.numel()] = values
        rows.append(row)
    if not rows:
        return None
    return torch.stack(rows[:needed])


def _metadata_train_path(config, model_dir: Path) -> Path | None:
    metadata = getattr(config, "training_metadata", {}) or {}
    raw_path = metadata.get("train_path")
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = model_dir / raw_path
    if candidate.exists():
        return candidate
    return None


def _write_summary(
    run_dir: Path,
    run_name: str,
    sample_files: list[Path],
    report: dict[str, Any],
) -> None:
    lines = [f"# {run_name}", "", "## Samples", ""]
    for path in sample_files:
        preview = path.read_text(encoding="utf-8").splitlines()[:5]
        lines.append(f"### {path.name}")
        lines.extend(f"- {line}" for line in preview if line)
        lines.append("")
    lines.extend(["## Denoise", ""])
    if report.get("status") != "ok":
        lines.append(f"Skipped: {report.get('reason', 'unknown')}")
    else:
        lines.append("| t | MSE | Token Acc | Maskable CE | Maskable Acc |")
        lines.append("| --: | --: | --: | --: | --: |")
        for row in report["time_points"]:
            lines.append(
                "| {t:.2f} | {mse:.4f} | {token_accuracy:.4f} | "
                "{maskable_token_ce:.4f} | {maskable_token_accuracy:.4f} |".format(**row)
            )
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(values.device)
    if not bool(mask.any().item()):
        return float("nan")
    return float(values[mask].float().mean().detach().cpu())


def _run_name(config, model_dir: Path) -> str:
    metadata = getattr(config, "training_metadata", {}) or {}
    name = metadata.get("name")
    if name:
        return str(name)
    return model_dir.name


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
