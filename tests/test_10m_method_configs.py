from __future__ import annotations

import unittest
from pathlib import Path

from babylm_elf.config import load_config
from babylm_elf.modeling.model import BabyLMELF


class TenMillionMethodConfigTest(unittest.TestCase):
    def test_ns1_denoise_configs(self) -> None:
        cases = [
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_4gpu.yml",
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_denoise_ns1_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_lr4e-4_denoise_ns1_4gpu",
                1.0,
            ),
            (
                "configs/2026_100M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_gb256_4gpu.yml",
                "configs/2026_100M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_gb256_denoise_ns1_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_gb256_denoise_ns1_4gpu",
                1.0,
            ),
        ]
        for base_path, ablation_path, expected_name, expected_scale in cases:
            with self.subTest(ablation_path=ablation_path):
                base = load_config(base_path)
                ablation = load_config(ablation_path)

                self.assertEqual(ablation.name, expected_name)
                self.assertEqual(
                    ablation.diffusion.denoiser_noise_scale,
                    expected_scale,
                )
                self.assertEqual(ablation.output_dir, base.output_dir)
                self.assertEqual(ablation.seed, base.seed)
                self.assertEqual(ablation.epochs, base.epochs)
                self.assertEqual(ablation.batch_size, base.batch_size)
                self.assertEqual(
                    ablation.gradient_accumulation_steps,
                    base.gradient_accumulation_steps,
                )
                self.assertEqual(
                    ablation.checkpoint_word_limit,
                    base.checkpoint_word_limit,
                )
                self.assertEqual(ablation.model, base.model)
                self.assertEqual(ablation.data, base.data)
                self.assertEqual(ablation.optim, base.optim)
                self.assertEqual(
                    ablation.diffusion.decoder_probability,
                    base.diffusion.decoder_probability,
                )
                self.assertEqual(
                    ablation.diffusion.mlm_mask_strategy,
                    base.diffusion.mlm_mask_strategy,
                )
                self.assertEqual(
                    ablation.diffusion.mlm_eval_adapter,
                    "mlm_mask_latent",
                )
                self.assertEqual(ablation.diffusion.mlm_mask_latent_seed, 0)
                self.assertEqual(ablation.diffusion.mlm_mask_latent_scale, 1.0)

    def test_cyclic_mask_mainline_configs(self) -> None:
        cases = [
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_cyclicmask_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_lr4e-4_cyclicmask_4gpu",
                2.0,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_denoise_ns1_cyclicmask_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_lr4e-4_denoise_ns1_cyclicmask_4gpu",
                1.0,
            ),
            (
                "configs/2026_100M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_gb256_cyclicmask_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_gb256_cyclicmask_4gpu",
                2.0,
            ),
            (
                "configs/2026_100M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_gb256_denoise_ns1_cyclicmask_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_gb256_denoise_ns1_cyclicmask_4gpu",
                1.0,
            ),
        ]
        for path, expected_name, expected_noise_scale in cases:
            with self.subTest(path=path):
                config = load_config(path)

                self.assertEqual(config.name, expected_name)
                self.assertEqual(config.diffusion.mlm_mask_schedule, "cyclic")
                self.assertEqual(config.diffusion.mlm_mask_seed, 0)
                self.assertEqual(config.diffusion.denoiser_noise_scale, expected_noise_scale)
                self.assertEqual(
                    config.diffusion.mlm_mask_strategy,
                    "one_per_segment_step10_then_step20",
                )
                self.assertEqual(config.diffusion.mlm_eval_adapter, "mlm_mask_latent")

    def test_current_10m_aux_lr_configs(self) -> None:
        cases = [
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_lr5e-4_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_lr5e-4_4gpu",
                0.0001,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_lr5e-4_aux2e-4_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_lr5e-4_aux2e-4_4gpu",
                0.0002,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_lr5e-4_aux3e-4_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_lr5e-4_aux3e-4_4gpu",
                0.0003,
            ),
        ]
        for (
            config_path,
            expected_name,
            aux_learning_rate,
        ) in cases:
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                self.assertEqual(config.name, expected_name)
                self.assertEqual(config.epochs, 10)
                self.assertEqual(config.batch_size, 32)
                self.assertEqual(config.gradient_accumulation_steps, 1)
                self.assertEqual(config.checkpoint_word_limit, 100_000_000)
                self.assertEqual(config.data.seq_length, 128)
                self.assertEqual(config.data.train_word_count, 10_000_000)
                self.assertEqual(config.model.embedding_size, 384)
                self.assertEqual(config.model.hidden_size, 512)
                self.assertEqual(config.model.intermediate_size, 2048)
                self.assertEqual(config.model.num_hidden_layers, 8)
                self.assertEqual(config.model.num_attention_heads, 8)
                self.assertEqual(config.model.bottleneck_size, 96)
                self.assertEqual(config.model.decoder_head_type, "bert_mlm_scaled")
                self.assertEqual(config.model.embedding_source, "learnable")
                self.assertEqual(config.optim.optimizer, "muon")
                self.assertEqual(config.optim.learning_rate, 0.0005)
                self.assertEqual(config.optim.min_lr, 0.00005)
                self.assertEqual(config.optim.aux_learning_rate, aux_learning_rate)
                self.assertEqual(config.diffusion.decoder_probability, 0.2)
                self.assertEqual(
                    config.diffusion.mlm_mask_strategy,
                    "one_per_segment_step10_then_step20",
                )
                self.assertTrue(config.diffusion.mlm_filter_punctuation_only)
                self.assertTrue(config.diffusion.mlm_filter_empty_control)
                self.assertEqual(config.diffusion.mlm_eval_adapter, "mlm_mask_latent")
                self.assertEqual(config.diffusion.mlm_mask_latent_seed, 0)
                self.assertEqual(config.diffusion.mlm_mask_latent_scale, 1.0)

    def test_ablations_config_directory_is_removed(self) -> None:
        self.assertFalse(Path("configs/ablations").exists())

    def test_no_public_scale2_routes(self) -> None:
        patterns = ["_denoise_" + "ns2", "AW4" + "ns2Cyc", "ns=2" + "/cyc"]
        roots = [
            Path("configs"),
            Path("scripts"),
            Path("tests"),
            Path("README.md"),
            Path("project.md"),
        ]
        matches: list[str] = []
        for root in roots:
            paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
            for path in paths:
                if path.suffix in {".pyc", ".pt", ".bin"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for pattern in patterns:
                    if pattern in text:
                        matches.append(f"{path}:{pattern}")
        self.assertEqual(matches, [])

    def test_current_10m_adamw_lr_sweep_configs(self) -> None:
        cases = [
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr2e-4_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_lr2e-4_4gpu",
                0.0002,
                0.00002,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr3e-4_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_lr3e-4_4gpu",
                0.0003,
                0.00003,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_lr4e-4_4gpu",
                0.0004,
                0.00004,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr5e-4_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_lr5e-4_4gpu",
                0.0005,
                0.00005,
            ),
        ]
        for config_path, expected_name, learning_rate, min_lr in cases:
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                self.assertEqual(config.name, expected_name)
                self.assertEqual(config.epochs, 10)
                self.assertEqual(config.batch_size, 32)
                self.assertEqual(config.gradient_accumulation_steps, 1)
                self.assertEqual(config.checkpoint_word_limit, 100_000_000)
                self.assertEqual(config.data.seq_length, 128)
                self.assertEqual(config.data.train_word_count, 10_000_000)
                self.assertEqual(config.model.embedding_size, 384)
                self.assertEqual(config.model.hidden_size, 512)
                self.assertEqual(config.model.intermediate_size, 2048)
                self.assertEqual(config.model.num_hidden_layers, 8)
                self.assertEqual(config.model.num_attention_heads, 8)
                self.assertEqual(config.model.bottleneck_size, 96)
                self.assertEqual(config.model.decoder_head_type, "bert_mlm_scaled")
                self.assertEqual(config.model.embedding_source, "learnable")
                self.assertEqual(config.optim.optimizer, "adamw")
                self.assertEqual(config.optim.learning_rate, learning_rate)
                self.assertIsNone(config.optim.aux_learning_rate)
                self.assertEqual(config.optim.min_lr, min_lr)
                self.assertEqual(config.optim.weight_decay, 0.0)
                self.assertEqual(config.optim.beta1, 0.9)
                self.assertEqual(config.optim.beta2, 0.999)
                self.assertEqual(config.optim.lr_schedule, "cosine")
                self.assertEqual(config.diffusion.decoder_probability, 0.2)
                self.assertEqual(
                    config.diffusion.mlm_mask_strategy,
                    "one_per_segment_step10_then_step20",
                )
                self.assertTrue(config.diffusion.mlm_filter_punctuation_only)
                self.assertTrue(config.diffusion.mlm_filter_empty_control)
                self.assertEqual(config.diffusion.mlm_eval_adapter, "mlm_mask_latent")
                self.assertEqual(config.diffusion.mlm_mask_latent_seed, 0)
                self.assertEqual(config.diffusion.mlm_mask_latent_scale, 1.0)

    def test_10m_adamw_official_optimizer_config(self) -> None:
        config = load_config(
            "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_official_lr4e-4_4gpu.yml"
        )

        self.assertEqual(
            config.name,
            "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_official_lr4e-4_4gpu",
        )
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertEqual(config.checkpoint_word_limit, 100_000_000)
        self.assertEqual(config.data.seq_length, 128)
        self.assertEqual(config.data.train_word_count, 10_000_000)
        self.assertEqual(config.model.embedding_size, 384)
        self.assertEqual(config.model.hidden_size, 512)
        self.assertEqual(config.model.intermediate_size, 2048)
        self.assertEqual(config.model.num_hidden_layers, 8)
        self.assertEqual(config.model.num_attention_heads, 8)
        self.assertEqual(config.model.bottleneck_size, 96)
        self.assertEqual(config.model.decoder_head_type, "bert_mlm_scaled")
        self.assertEqual(config.model.embedding_source, "learnable")
        self.assertEqual(config.optim.optimizer, "adamw")
        self.assertEqual(config.optim.learning_rate, 0.0004)
        self.assertIsNone(config.optim.aux_learning_rate)
        self.assertEqual(config.optim.weight_decay, 0.0)
        self.assertEqual(config.optim.beta1, 0.9)
        self.assertEqual(config.optim.beta2, 0.95)
        self.assertEqual(config.optim.eps, 0.00000001)
        self.assertEqual(config.optim.lr_schedule, "constant")
        self.assertEqual(config.optim.min_lr, 0.0)
        self.assertEqual(config.diffusion.decoder_probability, 0.2)
        self.assertEqual(config.diffusion.mlm_mask_strategy, "one_per_segment_step10_then_step20")
        self.assertTrue(config.diffusion.mlm_filter_punctuation_only)
        self.assertTrue(config.diffusion.mlm_filter_empty_control)

        model = BabyLMELF(config.model)
        param_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(param_count, 33_095_249)

    def test_10m_muon_official_lr_sweep_configs(self) -> None:
        cases = [
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr5e-3_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_official_lr5e-3_4gpu",
                0.005,
                0.005,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr1e-2_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_official_lr1e-2_4gpu",
                0.01,
                0.005,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr5e-2_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_official_lr5e-2_4gpu",
                0.05,
                0.005,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr5e-2_aux5e-4_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_official_lr5e-2_aux5e-4_4gpu",
                0.05,
                0.0005,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr5e-2_aux1e-3_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_official_lr5e-2_aux1e-3_4gpu",
                0.05,
                0.001,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr7e-2_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_official_lr7e-2_4gpu",
                0.07,
                0.005,
            ),
            (
                "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr1e-1_4gpu.yml",
                "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_official_lr1e-1_4gpu",
                0.1,
                0.005,
            ),
        ]
        for config_path, expected_name, learning_rate, aux_learning_rate in cases:
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                self.assertEqual(config.name, expected_name)
                self.assertEqual(config.epochs, 10)
                self.assertEqual(config.batch_size, 32)
                self.assertEqual(config.gradient_accumulation_steps, 1)
                self.assertEqual(config.checkpoint_word_limit, 100_000_000)
                self.assertEqual(config.data.seq_length, 128)
                self.assertEqual(config.data.train_word_count, 10_000_000)
                self.assertEqual(config.model.embedding_size, 384)
                self.assertEqual(config.model.hidden_size, 512)
                self.assertEqual(config.model.intermediate_size, 2048)
                self.assertEqual(config.model.num_hidden_layers, 8)
                self.assertEqual(config.model.num_attention_heads, 8)
                self.assertEqual(config.model.bottleneck_size, 96)
                self.assertEqual(config.model.decoder_head_type, "bert_mlm_scaled")
                self.assertEqual(config.model.embedding_source, "learnable")
                self.assertEqual(config.optim.optimizer, "muon")
                self.assertEqual(config.optim.learning_rate, learning_rate)
                self.assertEqual(config.optim.aux_learning_rate, aux_learning_rate)
                self.assertEqual(config.optim.weight_decay, 0.0)
                self.assertEqual(config.optim.beta1, 0.9)
                self.assertEqual(config.optim.beta2, 0.999)
                self.assertEqual(config.optim.eps, 0.00000001)
                self.assertEqual(config.optim.lr_schedule, "constant")
                self.assertEqual(config.optim.min_lr, 0.0)
                self.assertEqual(config.diffusion.decoder_probability, 0.2)
                self.assertEqual(
                    config.diffusion.mlm_mask_strategy,
                    "one_per_segment_step10_then_step20",
                )
                self.assertTrue(config.diffusion.mlm_filter_punctuation_only)
                self.assertTrue(config.diffusion.mlm_filter_empty_control)
                self.assertEqual(config.diffusion.mlm_eval_adapter, "mlm_mask_latent")
                self.assertEqual(config.diffusion.mlm_mask_latent_seed, 0)
                self.assertEqual(config.diffusion.mlm_mask_latent_scale, 1.0)

                model = BabyLMELF(config.model)
                param_count = sum(parameter.numel() for parameter in model.parameters())
                self.assertEqual(param_count, 33_095_249)

    def test_current_100m_mainline_config(self) -> None:
        config = load_config(
            "configs/2026_100M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_gb256_4gpu.yml"
        )

        self.assertEqual(
            config.name,
            "learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_gb256_4gpu",
        )
        self.assertEqual(config.output_dir, "outputs/2026_100M")
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.gradient_accumulation_steps, 2)
        self.assertEqual(config.checkpoint_word_limit, 1_000_000_000)
        self.assertEqual(config.data.hf_dataset, "BabyLM-community/BabyLM-2026-Strict")
        self.assertEqual(config.data.train_path, "data/2026_100M/tokenized/train_100M.bin")
        self.assertEqual(config.data.tokenizer_path, "data/2026_100M/tokenizer/tokenizer.json")
        self.assertEqual(config.data.manifest_path, "data/2026_100M/manifest.json")
        self.assertEqual(config.data.seq_length, 128)
        self.assertEqual(config.data.train_word_count, 100_000_000)
        self.assertEqual(config.model.embedding_size, 512)
        self.assertEqual(config.model.hidden_size, 768)
        self.assertEqual(config.model.intermediate_size, 3072)
        self.assertEqual(config.model.num_hidden_layers, 12)
        self.assertEqual(config.model.num_attention_heads, 12)
        self.assertEqual(config.model.bottleneck_size, 128)
        self.assertEqual(config.model.max_position_embeddings, 128)
        self.assertEqual(config.model.decoder_head_type, "bert_mlm_scaled")
        self.assertEqual(config.model.embedding_source, "learnable")
        self.assertEqual(config.optim.optimizer, "muon")
        self.assertEqual(config.optim.learning_rate, 0.00035)
        self.assertEqual(config.optim.aux_learning_rate, 0.0001)
        self.assertEqual(config.optim.min_lr, 0.000035)
        self.assertEqual(config.diffusion.decoder_probability, 0.2)
        self.assertEqual(config.diffusion.mlm_mask_strategy, "one_per_segment_step10_then_step20")
        self.assertTrue(config.diffusion.mlm_filter_punctuation_only)
        self.assertTrue(config.diffusion.mlm_filter_empty_control)


if __name__ == "__main__":
    unittest.main()
