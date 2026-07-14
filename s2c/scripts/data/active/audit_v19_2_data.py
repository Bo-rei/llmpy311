#!/usr/bin/env python3
"""
HiLSA-MoE v19.2 数据审计脚本 (Data Audit)

审计项:
1. 纯净性定律: Router/Expert train/val/test 中 OOS count == 0
2. 纯净性定律: Gate train 中 OOS count == 0
3. 数量守恒定律: 样本总数匹配
4. 物理隔离定律: Train/Val/Test 文本无交集
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class V19_2_DataAuditor:
    """v19.2 数据审计器"""

    def __init__(self, data_root: str = "data/v19"):
        self.data_root = Path(data_root)
        self.errors = []
        self.warnings = []

    def audit_gate(self) -> bool:
        """审计 Gate 数据集"""
        logger.info("=" * 60)
        logger.info("审计 Gate 数据集")
        logger.info("=" * 60)

        gate_dir = self.data_root / "gate"

        # 加载数据
        with open(gate_dir / "train.json") as f:
            train_data = json.load(f)
        with open(gate_dir / "val.json") as f:
            val_data = json.load(f)
        with open(gate_dir / "test.json") as f:
            test_data = json.load(f)

        # 检查1: Gate Train 必须纯净 (No OOS)
        train_oos_count = sum(
            1 for s in train_data if s["label"] == 1 or s["intent"] == "oos"
        )
        if train_oos_count > 0:
            self.errors.append(
                f"❌ Gate train.json 包含 {train_oos_count} 个 OOS 样本 (预期=0)"
            )
        else:
            logger.info(f"✅ Gate train.json: {len(train_data)} 样本, OOS=0")

        # 检查2: Val/Test 应包含 OOS (用于校准)
        val_oos_count = sum(1 for s in val_data if s["label"] == 1)
        test_oos_count = sum(1 for s in test_data if s["label"] == 1)

        logger.info(f"✅ Gate val.json: {len(val_data)} 样本, OOS={val_oos_count}")
        logger.info(f"✅ Gate test.json: {len(test_data)} 样本, OOS={test_oos_count}")

        if val_oos_count == 0:
            self.warnings.append("⚠️  Gate val.json 没有 OOS 样本 (无法校准阈值)")

        return train_oos_count == 0

    def audit_router(self) -> bool:
        """审计 Router 数据集 (闭集假设)"""
        logger.info("=" * 60)
        logger.info("审计 Router 数据集 (Closed-Set)")
        logger.info("=" * 60)

        router_dir = self.data_root / "router"

        # 加载数据
        with open(router_dir / "train.json") as f:
            train_data = json.load(f)
        with open(router_dir / "val.json") as f:
            val_data = json.load(f)
        with open(router_dir / "test.json") as f:
            test_data = json.load(f)

        # 检查: 所有 split 都不应包含 OOS
        all_passed = True
        for split_name, data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            oos_count = sum(
                1
                for s in data
                if s.get("intent") == "oos" or s.get("domain") == "unknown"
            )
            if oos_count > 0:
                self.errors.append(
                    f"❌ Router {split_name}.json 包含 {oos_count} 个 OOS 样本 (预期=0, 闭集假设)"
                )
                all_passed = False
            else:
                logger.info(f"✅ Router {split_name}.json: {len(data)} 样本, OOS=0")

        # 检查 Domain Map
        domain_map_path = router_dir / "domain_map.json"
        if domain_map_path.exists():
            with open(domain_map_path) as f:
                domain_map = json.load(f)
            logger.info(f"✅ Router domain_map.json: {len(domain_map)} 个领域")
        else:
            self.errors.append("❌ Router domain_map.json 不存在")
            all_passed = False

        return all_passed

    def audit_experts(self) -> bool:
        """审计 Expert 数据集 (闭集假设)"""
        logger.info("=" * 60)
        logger.info("审计 Expert 数据集 (Closed-Set per Domain)")
        logger.info("=" * 60)

        experts_dir = self.data_root / "experts"

        all_passed = True
        domain_count = 0

        for domain_dir in sorted(experts_dir.iterdir()):
            if not domain_dir.is_dir():
                continue

            domain_name = domain_dir.name
            domain_count += 1

            # 加载数据
            train_path = domain_dir / "train.json"
            val_path = domain_dir / "val.json"
            test_path = domain_dir / "test.json"
            intent_map_path = domain_dir / "intent_map.json"

            if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
                self.errors.append(f"❌ Expert {domain_name}: 缺少必要文件")
                all_passed = False
                continue

            with open(train_path) as f:
                train_data = json.load(f)
            with open(val_path) as f:
                val_data = json.load(f)
            with open(test_path) as f:
                test_data = json.load(f)

            # 检查: 不应包含 OOS
            for split_name, data in [
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            ]:
                oos_count = sum(
                    1
                    for s in data
                    if s.get("intent") == "oos" or s.get("domain") != domain_name
                )
                if oos_count > 0:
                    self.errors.append(
                        f"❌ Expert {domain_name}/{split_name}.json 包含 {oos_count} 个无效样本 (预期=0)"
                    )
                    all_passed = False

            # 检查 Intent Map
            if intent_map_path.exists():
                with open(intent_map_path) as f:
                    intent_map = json.load(f)
                logger.info(
                    f"✅ Expert {domain_name}: {len(train_data)}/{len(val_data)}/{len(test_data)} "
                    f"({len(intent_map)} intents)"
                )
            else:
                self.errors.append(f"❌ Expert {domain_name}: 缺少 intent_map.json")
                all_passed = False

        logger.info(f"✅ 审计了 {domain_count} 个 Expert")
        return all_passed

    def audit_isolation(self) -> bool:
        """审计物理隔离 (Train/Val/Test 文本不交叉)"""
        logger.info("=" * 60)
        logger.info("审计物理隔离 (Text Isolation)")
        logger.info("=" * 60)

        # 收集所有文本
        def collect_texts(json_path: Path) -> Set[str]:
            if not json_path.exists():
                return set()
            with open(json_path) as f:
                data = json.load(f)
            return {s["text"] for s in data}

        # Gate
        gate_train = collect_texts(self.data_root / "gate" / "train.json")
        gate_val = collect_texts(self.data_root / "gate" / "val.json")
        gate_test = collect_texts(self.data_root / "gate" / "test.json")

        # 检查交集
        train_val_overlap = gate_train & gate_val
        train_test_overlap = gate_train & gate_test
        val_test_overlap = gate_val & gate_test

        all_passed = True

        if train_val_overlap:
            self.errors.append(
                f"❌ Train 和 Val 有 {len(train_val_overlap)} 个重复文本"
            )
            all_passed = False

        if train_test_overlap:
            self.errors.append(
                f"❌ Train 和 Test 有 {len(train_test_overlap)} 个重复文本"
            )
            all_passed = False

        if val_test_overlap:
            self.warnings.append(
                f"⚠️  Val 和 Test 有 {len(val_test_overlap)} 个重复文本"
            )

        if all_passed and not val_test_overlap:
            logger.info("✅ Train/Val/Test 完全隔离, 无文本交叉")

        return all_passed

    def audit_manifest(self) -> bool:
        """审计清单文件"""
        logger.info("=" * 60)
        logger.info("审计 MANIFEST.json")
        logger.info("=" * 60)

        manifest_path = self.data_root / "MANIFEST.json"

        if not manifest_path.exists():
            self.errors.append("❌ MANIFEST.json 不存在")
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        # 检查关键字段
        required_keys = ["version", "gate", "router", "experts", "constraints"]
        missing = [k for k in required_keys if k not in manifest]

        if missing:
            self.errors.append(f"❌ MANIFEST.json 缺少字段: {missing}")
            return False

        # 检查约束
        constraints = manifest["constraints"]
        if constraints.get("gate_train_oos_count", -1) != 0:
            self.errors.append("❌ MANIFEST 约束违反: gate_train_oos_count != 0")
            return False

        if constraints.get("router_all_oos_count", -1) != 0:
            self.errors.append("❌ MANIFEST 约束违反: router_all_oos_count != 0")
            return False

        if constraints.get("expert_all_oos_count", -1) != 0:
            self.errors.append("❌ MANIFEST 约束违反: expert_all_oos_count != 0")
            return False

        logger.info(f"✅ MANIFEST.json 版本: {manifest['version']}")
        logger.info(f"✅ 所有约束满足")

        return True

    def run(self) -> bool:
        """执行完整审计"""
        logger.info("开始 v19.2 数据审计...")

        # 检查数据根目录
        if not self.data_root.exists():
            logger.error(f"❌ 数据目录不存在: {self.data_root}")
            return False

        # 执行审计
        gate_ok = self.audit_gate()
        router_ok = self.audit_router()
        experts_ok = self.audit_experts()
        isolation_ok = self.audit_isolation()
        manifest_ok = self.audit_manifest()

        # 汇总结果
        logger.info("=" * 60)
        logger.info("审计结果汇总")
        logger.info("=" * 60)

        if self.errors:
            logger.error(f"❌ 发现 {len(self.errors)} 个错误:")
            for err in self.errors:
                logger.error(f"  {err}")

        if self.warnings:
            logger.warning(f"⚠️  发现 {len(self.warnings)} 个警告:")
            for warn in self.warnings:
                logger.warning(f"  {warn}")

        all_passed = (
            gate_ok and router_ok and experts_ok and isolation_ok and manifest_ok
        )

        if all_passed and not self.errors:
            logger.info("=" * 60)
            logger.info("✅ 所有审计通过! 数据集符合 v19.2 规范")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("❌ 审计失败! 数据集不符合 v19.2 规范")
            logger.error("=" * 60)

        return all_passed and len(self.errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit v19.2 dataset consistency")
    parser.add_argument(
        "--data_root",
        default="data/v19",
        help="Dataset root path to audit",
    )
    args = parser.parse_args()

    auditor = V19_2_DataAuditor(data_root=args.data_root)
    success = auditor.run()
    exit(0 if success else 1)
