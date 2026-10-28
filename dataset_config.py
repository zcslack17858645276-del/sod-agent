"""数据集路径配置。

所有路径均以项目根目录为基准，通过 `from dataset_config import ...` 导入使用。

目录约定：
    datasets/
        RGB-T/<数据集>/{Train,Test}/<im>/<gt>/T      # 第二模态为热红外(T)
        RGB-D/<数据集>/{RGB,GT,depth}                 # 第二模态为深度(depth)

使用示例：
    from dataset_config import get_train_paths, get_test_sets

    im_root, gt_root, modality_root = get_train_paths()
    for name, cfg in get_test_sets():
        ...
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 根目录
# ---------------------------------------------------------------------------

# 项目根目录（本文件所在目录）
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据集根目录，可通过环境变量 SAM2_DATASET_ROOT 覆盖，便于迁移到其他机器
DATASETS_ROOT = Path(os.environ.get("SAM2_DATASET_ROOT", PROJECT_ROOT / "datasets"))

RGB_T_ROOT = DATASETS_ROOT / "RGB-T"
RGB_D_ROOT = DATASETS_ROOT / "RGB-D"

# 第二模态目录名：RGB-T 数据集用 T（热红外），RGB-D 数据集用 depth（深度）
T_DIR = "T"
MODALITY_DEPTH_DIR = "depth"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPaths:
    """一个数据划分下三个子目录的路径。"""

    im_root: Path
    gt_root: Path
    modality_root: Path

    def exists(self) -> bool:
        """三个目录是否都存在。"""
        return all(p.is_dir() for p in (self.im_root, self.gt_root, self.modality_root))

    def missing(self) -> List[Path]:
        """返回不存在的目录列表。"""
        return [p for p in (self.im_root, self.gt_root, self.modality_root) if not p.is_dir()]

    def as_dict(self) -> Dict[str, str]:
        """转换为字符串字典，便于日志打印或序列化。"""
        return {
            "im_root": str(self.im_root),
            "gt_root": str(self.gt_root),
            "modality_root": str(self.modality_root),
        }


@dataclass(frozen=True)
class DatasetSpec:
    """一个数据集的完整描述，包含 train / test 两个划分。"""

    name: str
    train: SplitPaths
    test: SplitPaths

    def split(self, split: str) -> SplitPaths:
        """按名称获取划分：`'train'` 或 `'test'`。"""
        if split == "train":
            return self.train
        if split == "test":
            return self.test
        raise ValueError(f"未知的数据划分: {split!r}（可选 'train' / 'test'）")


def _split(base: Path, im: str, gt: str, modality: str = MODALITY_DEPTH_DIR) -> SplitPaths:
    """按子目录名拼出一个划分的路径。

    `modality` 参数为第二模态目录名：RGB-T 数据集传 `T_DIR`，RGB-D 传 `MODALITY_DEPTH_DIR`。
    """
    return SplitPaths(
        im_root=base / im,
        gt_root=base / gt,
        modality_root=base / modality,
    )


def _single_split(base: Path, im: str = "RGB", gt: str = "GT", modality: str = MODALITY_DEPTH_DIR) -> SplitPaths:
    """没有 Train/Test 划分的数据集，train 与 test 共用同一份目录。"""
    return _split(base, im, gt, modality)


# ---------------------------------------------------------------------------
# 数据集注册表
# ---------------------------------------------------------------------------

DATASETS: Dict[str, DatasetSpec] = {
    # ==================== RGB-T 视频显著性数据集 ====================
    # 第二模态目录名为 T（热红外图像）
    "VT5000": DatasetSpec(
        name="VT5000",
        train=_split(RGB_T_ROOT / "VT5000" / "Train", "RGB", "GT", T_DIR),
        test=_split(RGB_T_ROOT / "VT5000" / "Test", "RGB", "GT", T_DIR),
    ),
    "UVT20K": DatasetSpec(
        name="UVT20K",
        # 训练集使用 im / gt 命名
        train=_split(RGB_T_ROOT / "UVT20K" / "Train", "im", "gt", T_DIR),
        # 测试集使用 RGB / GT 命名
        test=_split(RGB_T_ROOT / "UVT20K" / "Test", "RGB", "GT", T_DIR),
    ),
    "VT1000": DatasetSpec(
        name="VT1000",
        train=_single_split(RGB_T_ROOT / "VT1000", modality=T_DIR),
        test=_single_split(RGB_T_ROOT / "VT1000", modality=T_DIR),
    ),
    "VT821": DatasetSpec(
        name="VT821",
        train=_single_split(RGB_T_ROOT / "VT821", modality=T_DIR),
        test=_single_split(RGB_T_ROOT / "VT821", modality=T_DIR),
    ),
    "unaligned_VT5000": DatasetSpec(
        name="unaligned_VT5000",
        train=_single_split(RGB_T_ROOT / "unaligned_dataset" / "VT5000-Test_unalign", modality=T_DIR),
        test=_single_split(RGB_T_ROOT / "unaligned_dataset" / "VT5000-Test_unalign", modality=T_DIR),
    ),
    "unaligned_VT1000": DatasetSpec(
        name="unaligned_VT1000",
        train=_single_split(RGB_T_ROOT / "unaligned_dataset" / "VT1000_unalign", modality=T_DIR),
        test=_single_split(RGB_T_ROOT / "unaligned_dataset" / "VT1000_unalign", modality=T_DIR),
    ),
    "unaligned_VT821": DatasetSpec(
        name="unaligned_VT821",
        train=_single_split(RGB_T_ROOT / "unaligned_dataset" / "VT821_unalign", modality=T_DIR),
        test=_single_split(RGB_T_ROOT / "unaligned_dataset" / "VT821_unalign", modality=T_DIR),
    ),

    # ==================== RGB-D 图像显著性数据集 ====================
    # 第二模态目录名为 depth（深度图）
    "NJU2K": DatasetSpec(
        name="NJU2K",
        train=_single_split(RGB_D_ROOT / "NJU2K"),
        test=_single_split(RGB_D_ROOT / "NJU2K"),
    ),
    "NLPR": DatasetSpec(
        name="NLPR",
        train=_single_split(RGB_D_ROOT / "NLPR"),
        test=_single_split(RGB_D_ROOT / "NLPR"),
    ),
    "DUT-RGBD": DatasetSpec(
        name="DUT-RGBD",
        train=_single_split(RGB_D_ROOT / "DUT-RGBD"),
        test=_single_split(RGB_D_ROOT / "DUT-RGBD"),
    ),
    "DES": DatasetSpec(
        name="DES",
        train=_single_split(RGB_D_ROOT / "DES"),
        test=_single_split(RGB_D_ROOT / "DES"),
    ),
    "LFSD": DatasetSpec(
        name="LFSD",
        train=_single_split(RGB_D_ROOT / "LFSD"),
        test=_single_split(RGB_D_ROOT / "LFSD"),
    ),
    "SIP": DatasetSpec(
        name="SIP",
        train=_single_split(RGB_D_ROOT / "SIP"),
        test=_single_split(RGB_D_ROOT / "SIP"),
    ),
    "STERE": DatasetSpec(
        name="STERE",
        train=_single_split(RGB_D_ROOT / "STERE"),
        test=_single_split(RGB_D_ROOT / "STERE"),
    ),
    "train2": DatasetSpec(
        name="train2",
        train=_split(RGB_D_ROOT / "train2", "train_images", "train_masks", "train_depth"),
        test=_split(RGB_D_ROOT / "train2", "train_images", "train_masks", "train_depth"),
    ),
    "train_DUT": DatasetSpec(
        name="train_DUT",
        train=_split(RGB_D_ROOT / "train_DUT", "train_images", "train_masks", "train_depth"),
        test=_split(RGB_D_ROOT / "train_DUT", "train_images", "train_masks", "train_depth"),
    ),
}


# ---------------------------------------------------------------------------
# 当前启用的数据集（切换数据集只需修改这里）
# ---------------------------------------------------------------------------

# 训练集名称，需为 DATASETS 中的键
TRAIN_DATASET = "VT5000"

# 测试集名称列表，需为 DATASETS 中的键
TEST_DATASETS: List[str] = ["VT5000"]


# ---------------------------------------------------------------------------
# 访问接口
# ---------------------------------------------------------------------------


def _get_spec(dataset_name: str) -> DatasetSpec:
    if dataset_name not in DATASETS:
        raise KeyError(
            f"未知数据集 {dataset_name!r}，可选: {sorted(DATASETS)}"
        )
    return DATASETS[dataset_name]


def get_train_paths(dataset_name: Optional[str] = None, split: str = "train") -> Tuple[Path, Path, Path]:
    """返回训练集的 (im_root, gt_root, modality_root) 三个路径。"""
    spec = _get_spec(dataset_name or TRAIN_DATASET)
    paths = spec.split(split)
    return paths.im_root, paths.gt_root, paths.modality_root


def get_test_sets(dataset_names: Optional[Iterable[str]] = None) -> List[Dict[str, str]]:
    """返回测试集配置列表，每项为 {'name', 'im_root', 'gt_root', 'modality_root'}。"""
    names = list(dataset_names) if dataset_names is not None else list(TEST_DATASETS)
    test_sets: List[Dict[str, str]] = []
    for name in names:
        paths = _get_spec(name).split("test")
        test_sets.append({"name": name, **paths.as_dict()})
    return test_sets


def get_split(dataset_name: str, split: str = "train") -> SplitPaths:
    """获取指定数据集、指定划分的路径对象。"""
    return _get_spec(dataset_name).split(split)


def check_dataset(dataset_name: str, split: str = "train") -> bool:
    """检查数据集目录是否齐全，缺失时打印警告并返回 False。"""
    paths = _get_spec(dataset_name).split(split)
    missing = paths.missing()
    if missing:
        print(
            f"[Warning] 数据集 {dataset_name!r} ({split}) 缺少以下目录: "
            + ", ".join(str(p) for p in missing)
        )
        return False
    return True


def describe(dataset_names: Optional[Iterable[str]] = None) -> str:
    """返回所有（或指定）数据集的路径说明文本，便于打印排查。"""
    names = list(dataset_names) if dataset_names is not None else list(DATASETS)
    lines = []
    for name in names:
        spec = _get_spec(name)
        lines.append(f"[{name}]")
        for split in ("train", "test"):
            paths = spec.split(split)
            flag = "OK " if paths.exists() else "MISS"
            lines.append(f"  {split:<5} {flag} im={paths.im_root}")
            lines.append(f"        {'':<4} gt={paths.gt_root}")
            lines.append(f"        {'':<4} modality={paths.modality_root}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
