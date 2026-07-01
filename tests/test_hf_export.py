from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
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


@unittest.skipIf(AutoConfig is None, "transformers is not installed")
class HuggingFaceExportTest(unittest.TestCase):
    def test_export_supports_scratch_encoder_mlm_backend(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        tokenizer_path = project_root / "data/smoke/tokenizer/tokenizer.json"
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
            diffusion=DiffusionConfig(decoder_noise_scale=5.0),
        )
        model = BabyLMELF(model_config)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            export_dir = root / "hf"
            torch.save(
                {
                    "model": model.state_dict(),
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
            self.assertEqual(exported_config["evaluation_config"]["adapter"], "mlm_direct")
            self.assertEqual(exported_config["evaluation_config"]["mc_samples"], 1)
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
            logits = masked_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            self.assertEqual(tuple(logits.shape), (1, 4, 384))
            self.assertTrue(torch.isfinite(logits).all())

    def test_export_supports_babylm_encoder_and_mlm_interfaces(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        tokenizer_path = project_root / "data/smoke/tokenizer/tokenizer.json"
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
            diffusion=DiffusionConfig(decoder_noise_scale=5.0),
        )
        model = BabyLMELF(model_config)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            export_dir = root / "hf"
            torch.save({"model": model.state_dict()}, checkpoint_path)
            export_checkpoint_to_hf(checkpoint_path, export_dir, config)

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
                exported_config["diagnostic_generation_config"]["self_cond_cfg_scale"],
                3.0,
            )
            self.assertTrue((export_dir / "model.safetensors").exists())
            self.assertFalse((export_dir / "pytorch_model.bin").exists())

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
            self.assertTrue(torch.isfinite(encoder.babylm_elf.rope.cos).all())
            self.assertTrue(torch.isfinite(encoder.babylm_elf.rope.sin).all())
            torch.testing.assert_close(encoder.babylm_elf.rope.cos, model.rope.cos)
            torch.testing.assert_close(encoder.babylm_elf.rope.sin, model.rope.sin)
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
            hidden.sum().backward()
            self.assertIsNotNone(encoder.babylm_elf.token_embedding.weight.grad)

            masked_lm = AutoModelForMaskedLM.from_pretrained(
                export_dir,
                trust_remote_code=True,
            )
            self.assertTrue(
                all(torch.isfinite(parameter).all() for parameter in masked_lm.parameters())
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
            labels = torch.full_like(masked_ids, -100)
            labels[:, 2] = input_ids[:, 2]
            loss = masked_lm(
                input_ids=masked_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
            self.assertTrue(torch.isfinite(loss))
            torch.manual_seed(0)
            ode_ids = masked_lm.generate(
                batch_size=2,
                sequence_length=5,
                num_steps=2,
                sampling_method="ode",
                time_schedule="uniform",
                self_cond_cfg_scale=1.0,
            )
            torch.manual_seed(0)
            sde_ids = masked_lm.generate(
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
            torch.save(
                {"model": model.state_dict(), "metadata": {"target_words": 1_000_000}},
                required_dir / "chck_1M.pt",
            )
            torch.save(
                {"model": model.state_dict(), "metadata": {"target_words": 2_000_000}},
                required_dir / "chck_2M.pt",
            )
            revisions_dir = root / "revisions"
            export_all_revisions(run_dir, revisions_dir, config)
            self.assertTrue((revisions_dir / "hf_export_test_hf/model.safetensors").exists())
            self.assertTrue((revisions_dir / "chck_1M/model.safetensors").exists())
            self.assertTrue((revisions_dir / "chck_2M/model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
