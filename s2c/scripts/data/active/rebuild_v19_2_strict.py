#!/usr/bin/env python3
"""
HiLSA-MoE v19.2 数据重构脚本 (Strict Data Rebuild)

核心原则:
1. Gate: train=Known only, val/test=Known+OOS
2. Router: train/val/test=Known only (闭集假设)
3. Expert: train/val/test=Known only (闭集假设)

物理隔离 + 角色对齐
"""

import json
import logging
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import hashlib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class V19_2_DataBuilder:
    """v19.2 严格数据构建器"""

    def __init__(
        self,
        raw_data_path: str = "clinc_data_origin/data/data_full.json",
        known_intents_path: str = "data/v19/KNOWN_INTENTS.json",
        output_root: str = "data/v19",
        seed: int = 42,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.known_intents_path = Path(known_intents_path)
        self.output_root = Path(output_root)
        self.seed = seed

        random.seed(seed)

        # 加载原始数据
        logger.info(f"加载原始数据: {self.raw_data_path}")
        with open(self.raw_data_path) as f:
            self.raw_data = json.load(f)

        # 加载已知意图列表
        logger.info(f"加载已知意图: {self.known_intents_path}")
        with open(self.known_intents_path) as f:
            known_cfg = json.load(f)
            self.ratio = float(known_cfg.get("ratio", 0.0))
            self.known_intents = set(known_cfg["known_intents"])
            self.unknown_intents = set(known_cfg["unknown_intents"])

        # Domain 映射 (CLINC150 标准)
        self.domain_map = {
            "banking": 0,
            "credit_cards": 1,
            "work": 2,
            "home": 3,
            "kitchen_and_dining": 4,
            "travel": 5,
            "utility": 6,
            "auto_and_commute": 7,
            "small_talk": 8,
            "meta": 9,
        }

        # 意图到领域的映射 (需从原始数据构建)
        self.intent_to_domain = self._build_intent_domain_map()

    def _build_intent_domain_map(self) -> Dict[str, str]:
        """从 domains.json 构建 intent -> domain 映射"""
        domains_path = self.raw_data_path.parent / "domains.json"

        if not domains_path.exists():
            logger.error(f"domains.json 不存在: {domains_path}")
            return {}

        with open(domains_path) as f:
            domains_data = json.load(f)

        mapping = {}
        for domain_name, intents in domains_data.items():
            for intent in intents:
                mapping[intent] = domain_name

        logger.info(f"构建意图-领域映射: {len(mapping)} 个意图")
        return mapping

    def _hash_text(self, text: str) -> str:
        """计算文本SHA256哈希"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def build_gate_datasets(self):
        """构建 Gate 数据集 (唯一拒识源)"""
        logger.info("=" * 60)
        logger.info("构建 Gate 数据集")
        logger.info("=" * 60)

        gate_dir = self.output_root / "gate"
        gate_dir.mkdir(parents=True, exist_ok=True)

        # Train: Pure Known (SVDD 纯净假设)
        train_samples = []
        for text, intent in self.raw_data["train"]:
            if intent in self.known_intents:
                domain = self.intent_to_domain.get(intent, "unknown")
                train_samples.append(
                    {
                        "text": text,
                        "intent": intent,
                        "domain": domain,
                        "split": "train",
                        "label": 0,  # 0=Known (SVDD 假设所有训练样本都是 Known)
                    }
                )

        # Val: Known + Real OOS (校准用)
        val_samples = []
        for text, intent in self.raw_data["val"]:
            if intent in self.known_intents:
                domain = self.intent_to_domain.get(intent, "unknown")
                val_samples.append(
                    {
                        "text": text,
                        "intent": intent,
                        "domain": domain,
                        "split": "val",
                        "label": 0,  # Known
                    }
                )

        # 添加 OOS Val
        for text, intent in self.raw_data.get("oos_val", []):
            val_samples.append(
                {
                    "text": text,
                    "intent": "oos",
                    "domain": "unknown",
                    "split": "oos_val",
                    "label": 1,  # 1=OOS
                }
            )

        # Test: Known + Unknown + Official OOS
        test_samples = []
        for text, intent in self.raw_data["test"]:
            domain = self.intent_to_domain.get(intent, "unknown")
            if intent in self.known_intents:
                test_samples.append(
                    {
                        "text": text,
                        "intent": intent,
                        "domain": domain,
                        "split": "test",
                        "label": 0,  # Known
                    }
                )
            elif intent in self.unknown_intents:
                test_samples.append(
                    {
                        "text": text,
                        "intent": intent,
                        "domain": "unknown",
                        "split": "test",
                        "label": 1,  # OOS (Unknown Intent)
                    }
                )

        # 添加 Official OOS Test
        for text, intent in self.raw_data.get("oos_test", []):
            test_samples.append(
                {
                    "text": text,
                    "intent": "oos",
                    "domain": "unknown",
                    "split": "oos_test",
                    "label": 1,  # OOS
                }
            )

        # 保存
        with open(gate_dir / "train.json", "w") as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)
        with open(gate_dir / "val.json", "w") as f:
            json.dump(val_samples, f, indent=2, ensure_ascii=False)
        with open(gate_dir / "test.json", "w") as f:
            json.dump(test_samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Gate Train: {len(train_samples)} (Pure Known)")
        logger.info(f"Gate Val: {len(val_samples)} (Known + OOS)")
        logger.info(f"Gate Test: {len(test_samples)} (Known + Unknown + OOS)")

        return {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        }

    def build_router_datasets(self):
        """构建 Router 数据集 (闭集假设 - Known Only)"""
        logger.info("=" * 60)
        logger.info("构建 Router 数据集 (Closed-Set)")
        logger.info("=" * 60)

        router_dir = self.output_root / "router"
        router_dir.mkdir(parents=True, exist_ok=True)

        def build_split(split_name: str):
            samples = []
            for text, intent in self.raw_data[split_name]:
                # 严格过滤: 只保留 Known
                if intent in self.known_intents:
                    domain = self.intent_to_domain.get(intent, "unknown")
                    samples.append(
                        {
                            "text": text,
                            "intent": intent,
                            "domain": domain,
                            "split": split_name,
                            "label": self.domain_map[domain],  # Domain ID (0-9)
                        }
                    )
            return samples

        train_samples = build_split("train")
        val_samples = build_split("val")
        test_samples = build_split("test")

        # 保存
        with open(router_dir / "train.json", "w") as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)
        with open(router_dir / "val.json", "w") as f:
            json.dump(val_samples, f, indent=2, ensure_ascii=False)
        with open(router_dir / "test.json", "w") as f:
            json.dump(test_samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Router Train: {len(train_samples)} (Known Only)")
        logger.info(f"Router Val: {len(val_samples)} (Known Only)")
        logger.info(f"Router Test: {len(test_samples)} (Known Only)")

        # 保存 Domain Map
        with open(router_dir / "domain_map.json", "w") as f:
            json.dump(self.domain_map, f, indent=2)

        return {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        }

    def build_expert_datasets(self):
        """构建 Expert 数据集 (闭集假设 - 每个领域独立)"""
        logger.info("=" * 60)
        logger.info("构建 Expert 数据集 (Closed-Set per Domain)")
        logger.info("=" * 60)

        experts_dir = self.output_root / "experts"
        experts_dir.mkdir(parents=True, exist_ok=True)

        stats = {}

        for domain_name, domain_id in self.domain_map.items():
            domain_dir = experts_dir / domain_name
            domain_dir.mkdir(parents=True, exist_ok=True)

            # 获取该领域的已知意图
            domain_intents = [
                intent
                for intent, domain in self.intent_to_domain.items()
                if domain == domain_name and intent in self.known_intents
            ]

            # 构建 Local Intent ID 映射
            local_intent_map = {
                intent: idx for idx, intent in enumerate(sorted(domain_intents))
            }

            def build_domain_split(split_name: str):
                samples = []
                for text, intent in self.raw_data[split_name]:
                    domain = self.intent_to_domain.get(intent, "unknown")
                    if domain == domain_name and intent in self.known_intents:
                        samples.append(
                            {
                                "text": text,
                                "intent": intent,
                                "domain": domain,
                                "split": split_name,
                                "label": local_intent_map[intent],  # Local Intent ID
                            }
                        )
                return samples

            train_samples = build_domain_split("train")
            val_samples = build_domain_split("val")
            test_samples = build_domain_split("test")

            # 保存
            with open(domain_dir / "train.json", "w") as f:
                json.dump(train_samples, f, indent=2, ensure_ascii=False)
            with open(domain_dir / "val.json", "w") as f:
                json.dump(val_samples, f, indent=2, ensure_ascii=False)
            with open(domain_dir / "test.json", "w") as f:
                json.dump(test_samples, f, indent=2, ensure_ascii=False)

            # 保存 Intent Map
            with open(domain_dir / "intent_map.json", "w") as f:
                json.dump(local_intent_map, f, indent=2)

            stats[domain_name] = {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
                "num_intents": len(local_intent_map),
            }

            logger.info(
                f"  {domain_name}: {len(train_samples)}/{len(val_samples)}/{len(test_samples)} "
                f"({len(local_intent_map)} intents)"
            )

        return stats

    def generate_manifest(self, gate_stats, router_stats, expert_stats):
        """生成数据集清单"""
        manifest = {
            "version": "v19.2",
            "timestamp": "2026-01-29",
            "seed": self.seed,
            "known_intents_count": len(self.known_intents),
            "unknown_intents_count": len(self.unknown_intents),
            "gate": gate_stats,
            "router": router_stats,
            "experts": expert_stats,
            "constraints": {
                "gate_train_oos_count": 0,
                "router_all_oos_count": 0,
                "expert_all_oos_count": 0,
            },
        }

        manifest_path = self.output_root / "MANIFEST.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Preserve the ratio-specific intent manifest for downstream gate training.
        known_copy_path = self.output_root / "KNOWN_INTENTS.json"
        with open(known_copy_path, "w") as f:
            json.dump(
                {
                    "ratio": self.ratio,
                    "seed": self.seed,
                    "known_intents": sorted(list(self.known_intents)),
                    "unknown_intents": sorted(list(self.unknown_intents)),
                    "known_count": len(self.known_intents),
                    "unknown_count": len(self.unknown_intents),
                    "expected_train_samples": manifest["gate"]["train"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"清单已生成: {manifest_path}")
        return manifest

    def run(self):
        """执行完整重构"""
        logger.info("开始 v19.2 数据重构...")

        # 创建输出目录
        self.output_root.mkdir(parents=True, exist_ok=True)

        # 构建三套数据集
        gate_stats = self.build_gate_datasets()
        router_stats = self.build_router_datasets()
        expert_stats = self.build_expert_datasets()

        # 生成清单
        manifest = self.generate_manifest(gate_stats, router_stats, expert_stats)

        logger.info("=" * 60)
        logger.info("✅ v19.2 数据重构完成!")
        logger.info(f"输出目录: {self.output_root}")
        logger.info("=" * 60)

        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build strict v19.2 datasets")
    parser.add_argument(
        "--raw_data_path",
        default="clinc_data_origin/data/data_full.json",
        help="Path to CLINC raw data_full.json",
    )
    parser.add_argument(
        "--known_intents_path",
        default="data/v19/KNOWN_INTENTS.json",
        help="Path to known/unknown intent config JSON",
    )
    parser.add_argument(
        "--output_root",
        default="data/v19",
        help="Output dataset root",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic build behavior",
    )
    args = parser.parse_args()

    builder = V19_2_DataBuilder(
        raw_data_path=args.raw_data_path,
        known_intents_path=args.known_intents_path,
        output_root=args.output_root,
        seed=args.seed,
    )
    manifest = builder.run()

    print("\n📊 数据集统计:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
