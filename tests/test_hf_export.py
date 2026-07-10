from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

os.environ.setdefault("HF_HOME", "/tmp/babylm-elf-hf-test")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/babylm-elf-hf-test/transformers")

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise
    AutoConfig = None
    AutoModel = None
    AutoModelForMaskedLM = None
    AutoTokenizer = None

from babylm_elf.cli.export_hf import export_all_revisions
from babylm_elf.config import DataConfig, DiffusionConfig, TrainConfig
from babylm_elf.export.convert_checkpoint import export_checkpoint_to_hf
from babylm_elf.modeling.model import BabyLMELF, BabyLMELFConfig
from babylm_elf.submission.contract import required_revisions, words_for_revision


def _test_tokenizer_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    smoke_path = project_root / "data/smoke/tokenizer/tokenizer.json"
    if smoke_path.exists():
        return smoke_path
    return project_root / "data/2026_10M/tokenizer/tokenizer.json"


@unittest.skipIf(AutoConfig is None, "transformers is not installed")
class HuggingFaceExportTest(unittest.TestCase):
    def test_export_supports_scratch_encoder_mlm_backend(self) -> None:
        tokenizer_path = _test_tokenizer_path()
        model_config = BabyLMELFConfig(
            vocab_size=384,
            base_vocab_size=384,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            bottleneck_size=8,
            embedding_source="scratch_t5_encoder",
            encoder_vocab_size=484,
            sentinel_start_id=384,
            sentinel_count=100,
            encoder_d_ff=32,
            encoder_d_kv=4,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_dropout_rate=0.0,
        )
        config = TrainConfig(
            name="scratch_export_test",
            word_exposure_offset=30_000_000,
            model=model_config,
            data=DataConfig(
                tokenizer_path=str(tokenizer_path),
                tokenizer_vocab_size=384,
                seq_length=16,
            ),
            diffusion=DiffusionConfig(),
        )
        model = BabyLMELF(model_config)
        ema_state = model.state_dict()
        raw_state = {
            name: tensor.detach().clone()
            for name, tensor in ema_state.items()
        }
        raw_state["codebook.bias"].add_(1.0)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            export_dir = root / "hf_scratch_export"
            torch.save(
                {
                    "model": ema_state,
                    "model_ema": ema_state,
                    "model_raw": raw_state,
                    "metadata": {
                        "words_seen": 100_000_000,
                        "target_words": 100_000_000,
                    },
                },
                checkpoint_path,
            )
            export_checkpoint_to_hf(checkpoint_path, export_dir, config)

            exported_config = json.loads(
                (export_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exported_config["evaluation_config"]["backend"], "mlm")
            self.assertEqual(
                exported_config["evaluation_config"]["adapter"],
                "mlm_mask_latent",
            )
            self.assertEqual(exported_config["evaluation_config"]["mc_samples"], 1)
            self.assertEqual(exported_config["evaluation_config"]["mask_latent_seed"], 0)
            self.assertEqual(exported_config["evaluation_config"]["mask_latent_scale"], 1.0)
            self.assertEqual(
                exported_config["babylm_elf_config"]["embedding_source"],
                "scratch_t5_encoder",
            )
            self.assertIsNone(
                exported_config["babylm_elf_config"]["encoder_checkpoint_path"]
            )
            self.assertIsNone(exported_config["babylm_elf_config"]["latent_stats_path"])
            self.assertEqual(exported_config["vocab_size"], 384)
            self.assertEqual(
                exported_config["training_metadata"]["word_exposure_offset"],
                30_000_000,
            )
            self.assertEqual(
                exported_config["training_metadata"]["checkpoint_metadata"]["words_seen"],
                100_000_000,
            )
            self.assertEqual(
                exported_config["training_metadata"]["weights_variant"],
                "ema",
            )
            self.assertTrue((export_dir / "mask_latent.py").exists())

            raw_export_dir = root / "hf_raw"
            export_checkpoint_to_hf(
                checkpoint_path,
                raw_export_dir,
                config,
                weights="raw",
            )
            raw_exported_config = json.loads(
                (raw_export_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                raw_exported_config["training_metadata"]["weights_variant"],
                "raw",
            )
            ema_exported_state = load_file(export_dir / "model.safetensors")
            raw_exported_state = load_file(raw_export_dir / "model.safetensors")
            self.assertFalse(
                torch.equal(
                    ema_exported_state["babylm_elf.codebook.bias"],
                    raw_exported_state["babylm_elf.codebook.bias"],
                )
            )

            tokenizer = AutoTokenizer.from_pretrained(
                export_dir,
                trust_remote_code=True,
            )
            input_ids = torch.tensor([[1, 7, tokenizer.mask_token_id, 2]])
            attention_mask = torch.ones_like(input_ids)

            encoder = AutoModel.from_pretrained(
                export_dir,
                trust_remote_code=True,
            )
            hidden = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            self.assertEqual(tuple(hidden.shape), (1, 4, 32))
            self.assertTrue(torch.isfinite(hidden).all())
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in encoder.babylm_elf.scratch_encoder.parameters()
                )
            )

            masked_lm = AutoModelForMaskedLM.from_pretrained(
                export_dir,
                trust_remote_code=True,
            )
            unmasked_ids = torch.tensor([[1, 7, 8, 2]])
            logits = masked_lm(
                input_ids=unmasked_ids,
                attention_mask=attention_mask,
            ).logits
            self.assertEqual(tuple(logits.shape), (1, 4, 384))
            self.assertTrue(torch.isfinite(logits).all())
            with self.assertRaisesRegex(ValueError, "mlm_mask_latent requires"):
                masked_lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

    def test_export_supports_babylm_encoder_and_mlm_interfaces(self) -> None:
        tokenizer_path = _test_tokenizer_path()
        model_config = BabyLMELFConfig(
            vocab_size=384,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            bottleneck_size=8,
        )
        config = TrainConfig(
            name="hf_export_test",
            model=model_config,
            data=DataConfig(
                tokenizer_path=str(tokenizer_path),
                tokenizer_vocab_size=384,
                seq_length=16,
            ),
            diffusion=DiffusionConfig(),
        )
        model = BabyLMELF(model_config)
        with torch.no_grad():
            model.mlm_mask_latent.copy_(
                torch.linspace(-0.5, 0.5, model_config.embedding_size)
            )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            export_dir = root / "hf_main_export"
            torch.save({"model": model.state_dict()}, checkpoint_path)
            export_checkpoint_to_hf(checkpoint_path, export_dir, config)
            with self.assertRaisesRegex(KeyError, "no 'model_raw'"):
                export_checkpoint_to_hf(
                    checkpoint_path,
                    root / "hf_raw_missing",
                    config,
                    weights="raw",
                )

            exported_config = json.loads(
                (export_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exported_config["hidden_size"], 32)
            self.assertEqual(exported_config["mask_token_id"], 4)
            self.assertEqual(
                exported_config["diagnostic_generation_config"]["sampling_method"],
                "sde",
            )
            self.assertEqual(
                exported_config["diagnostic_generation_config"]["purpose"],
                "optional_debug_only_not_babylm_official_scoring",
            )
            self.assertEqual(
                exported_config["diagnostic_generation_config"]["self_cond_cfg_scale"],
                3.0,
            )
            self.assertTrue((export_dir / "model.safetensors").exists())
            self.assertFalse((export_dir / "pytorch_model.bin").exists())
            self.assertTrue((export_dir / "mask_latent.py").exists())
            self.assertTrue((export_dir / "codebook.py").exists())
            self.assertTrue((export_dir / "positions.py").exists())
            fresh_cache = root / "fresh_hf_modules_cache"
            fresh_home = root / "fresh_hf_home"
            smoke = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import torch\n"
                        "from transformers import AutoModelForMaskedLM\n"
                        f"path = {str(export_dir)!r}\n"
                        "model = AutoModelForMaskedLM.from_pretrained("
                        "path, trust_remote_code=True)\n"
                        "input_ids = torch.tensor([[1, 7, 4, 2]])\n"
                        "logits = model("
                        "input_ids=input_ids, attention_mask=torch.ones_like(input_ids)"
                        ").logits\n"
                        "print(tuple(logits.shape))\n"
                    ),
                ],
                check=True,
                capture_output=True,
                env={
                    **os.environ,
                    "HF_HOME": str(fresh_home),
                    "HF_MODULES_CACHE": str(fresh_cache),
                },
                text=True,
            )
            self.assertIn("(1, 4, 384)", smoke.stdout)
            self.assertTrue(
                any(path.name == "mask_latent.py" for path in fresh_cache.rglob("mask_latent.py"))
            )

            tokenizer = AutoTokenizer.from_pretrained(
                export_dir,
                trust_remote_code=True,
            )
            self.assertEqual(tokenizer.mask_token_id, 4)
            self.assertEqual(tokenizer.pad_token_id, 3)
            self.assertEqual(tokenizer.model_max_length, 32)

            auto_config = AutoConfig.from_pretrained(
                export_dir,
                trust_remote_code=True,
            )
            self.assertEqual(auto_config.hidden_size, 32)
            self.assertEqual(
                auto_config.diagnostic_generation_config["sde_gamma"],
                1.0,
            )

            encoder = AutoModel.from_pretrained(
                export_dir,
                trust_remote_code=True,
            )
            self.assertTrue(
                all(torch.isfinite(parameter).all() for parameter in encoder.parameters())
            )
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    encoder.babylm_elf.state_dict()[name],
                    value,
                )
            input_ids = torch.tensor([[1, 7, 8, 2]])
            attention_mask = torch.ones_like(input_ids)
            direct_embeddings = model.embed_tokens(input_ids)
            direct_hidden = model.forward_hidden(
                torch.cat((direct_embeddings, torch.zeros_like(direct_embeddings)), dim=-1),
                torch.ones(1),
                attention_mask=attention_mask,
                self_cond_cfg_scale=torch.ones(1),
                decoder_step_active=torch.ones(1),
            )
            self.assertTrue(torch.isfinite(direct_hidden).all())
            loaded_embeddings = encoder.babylm_elf.embed_tokens(input_ids)
            torch.testing.assert_close(loaded_embeddings, direct_embeddings)
            self.assertTrue(
                torch.isfinite(encoder.babylm_elf.position_attention.cos).all()
            )
            self.assertTrue(
                torch.isfinite(encoder.babylm_elf.position_attention.sin).all()
            )
            torch.testing.assert_close(
                encoder.babylm_elf.position_attention.cos,
                model.position_attention.cos,
            )
            torch.testing.assert_close(
                encoder.babylm_elf.position_attention.sin,
                model.position_attention.sin,
            )
            loaded_direct_hidden = encoder.babylm_elf.forward_hidden(
                torch.cat((loaded_embeddings, torch.zeros_like(loaded_embeddings)), dim=-1),
                torch.ones(1),
                attention_mask=attention_mask,
                self_cond_cfg_scale=torch.ones(1),
                decoder_step_active=torch.ones(1),
            )
            torch.testing.assert_close(loaded_direct_hidden, direct_hidden)
            cfg_three_hidden = encoder.babylm_elf.forward_hidden(
                torch.cat((loaded_embeddings, torch.zeros_like(loaded_embeddings)), dim=-1),
                torch.ones(1),
                attention_mask=attention_mask,
                self_cond_cfg_scale=torch.full((1,), 3.0),
                decoder_step_active=torch.ones(1),
            )
            self.assertFalse(torch.allclose(cfg_three_hidden, loaded_direct_hidden))
            hidden = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            self.assertEqual(tuple(hidden.shape), (1, 4, 32))
            self.assertTrue(torch.isfinite(hidden).all())
            left_padded_ids = torch.tensor([[3, 3, 1, 7, 8, 2]])
            left_padded_mask = left_padded_ids.ne(tokenizer.pad_token_id).long()
            left_padded_hidden = encoder(
                input_ids=left_padded_ids,
                attention_mask=left_padded_mask,
            ).last_hidden_state
            torch.testing.assert_close(hidden, left_padded_hidden[:, -4:])
            hidden.sum().backward()
            self.assertIsNotNone(encoder.babylm_elf.token_embedding.weight.grad)

            masked_lm = AutoModelForMaskedLM.from_pretrained(
                export_dir,
                trust_remote_code=True,
            )
            self.assertTrue(
                all(torch.isfinite(parameter).all() for parameter in masked_lm.parameters())
            )
            torch.testing.assert_close(
                masked_lm.babylm_elf.mlm_mask_latent,
                model.mlm_mask_latent,
            )
            masked_ids = torch.tensor([[1, 7, tokenizer.mask_token_id, 2]])
            first = masked_lm(
                input_ids=masked_ids,
                attention_mask=attention_mask,
            ).logits
            second = masked_lm(
                input_ids=masked_ids,
                attention_mask=attention_mask,
            ).logits
            self.assertEqual(tuple(first.shape), (1, 4, 384))
            self.assertTrue(torch.isfinite(first).all())
            torch.testing.assert_close(first, second)

            with patch.dict(
                masked_lm.config.evaluation_config,
                {"adapter": "mlm_direct"},
            ):
                with self.assertRaisesRegex(ValueError, "only supports"):
                    masked_lm(
                        input_ids=masked_ids,
                        attention_mask=attention_mask,
                    )

            with patch.dict(
                masked_lm.config.evaluation_config,
                {"mask_latent_seed": 1},
            ):
                seed_metadata_ignored = masked_lm(
                    input_ids=masked_ids,
                    attention_mask=attention_mask,
                ).logits

            self.assertEqual(tuple(seed_metadata_ignored.shape), (1, 4, 384))
            self.assertTrue(torch.isfinite(seed_metadata_ignored).all())
            torch.testing.assert_close(first, seed_metadata_ignored)

            with patch.dict(
                "os.environ",
                {"BABYLM_ELF_CONTEXT_ABLATION": "zero_visible"},
            ):
                zero_visible = masked_lm(
                    input_ids=masked_ids,
                    attention_mask=attention_mask,
                ).logits

            self.assertEqual(tuple(zero_visible.shape), (1, 4, 384))
            self.assertTrue(torch.isfinite(zero_visible).all())
            self.assertGreater(
                float((first - zero_visible).detach().abs().max()),
                0.0,
            )
            labels = torch.full_like(masked_ids, -100)
            labels[:, 2] = input_ids[:, 2]
            loss = masked_lm(
                input_ids=masked_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(hasattr(masked_lm, "diagnostic_generate"))
            torch.manual_seed(0)
            ode_ids = masked_lm.diagnostic_generate(
                batch_size=2,
                sequence_length=5,
                num_steps=2,
                sampling_method="ode",
                time_schedule="uniform",
                self_cond_cfg_scale=1.0,
            )
            torch.manual_seed(0)
            sde_ids = masked_lm.diagnostic_generate(
                batch_size=2,
                sequence_length=5,
                num_steps=2,
                sampling_method="sde",
                time_schedule="uniform",
                self_cond_cfg_scale=3.0,
                sde_gamma=1.0,
            )
            self.assertEqual(tuple(ode_ids.shape), (2, 5))
            self.assertEqual(tuple(sde_ids.shape), (2, 5))
            self.assertTrue(torch.isfinite(ode_ids.float()).all())
            self.assertTrue(torch.isfinite(sde_ids.float()).all())
            self.assertTrue(ode_ids.ge(0).all())
            self.assertTrue(sde_ids.lt(model_config.vocab_size).all())

            run_dir = root / "run"
            checkpoint_dir = run_dir / "checkpoints"
            required_dir = checkpoint_dir / "babylm_required"
            required_dir.mkdir(parents=True)
            torch.save({"model": model.state_dict()}, checkpoint_dir / "final.pt")
            for revision in required_revisions("strict-small"):
                torch.save(
                    {
                        "model": model.state_dict(),
                        "metadata": {"target_words": words_for_revision(revision)},
                    },
                    required_dir / f"{revision}.pt",
                )
            revisions_dir = root / "revisions"
            export_all_revisions(
                run_dir,
                revisions_dir,
                config,
                track="strict-small",
            )
            self.assertTrue((revisions_dir / "hf_export_test_hf/model.safetensors").exists())
            self.assertTrue((revisions_dir / "chck_1M/model.safetensors").exists())
            self.assertTrue((revisions_dir / "chck_2M/model.safetensors").exists())

if __name__ == "__main__":
    unittest.main()
