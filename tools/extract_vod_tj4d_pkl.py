"""Extracted VoD/TJ4D dataset pkl generation pipeline.

This file centralizes the pkl-related generation logic that is otherwise
scattered across `tools/create_data.py`, `tools/data_converter/tj4d_converter.py`
and `tools/data_converter/create_gt_database.py`.
"""

import argparse
from os import path as osp

from tools.data_converter import tj4d_converter as tj4d
from tools.data_converter.create_gt_database import create_groundtruth_database


def generate_vod_tj4d_infos(root_path: str, info_prefix: str) -> None:
    """Generate *_infos_{train|val|trainval|test}.pkl and reduced point cloud.

    Args:
        root_path (str): Dataset root path.
        info_prefix (str): Prefix of output pkl names.
    """
    # Same core flow as `kitti_data_prep` in tools/create_data.py for VoD/TJ4D.
    tj4d.create_kitti_info_file(root_path, info_prefix)
    tj4d.create_reduced_point_cloud(root_path, info_prefix)


def generate_vod_tj4d_gt_database(
    root_path: str,
    info_prefix: str,
    out_dir: str,
    version: str = "v1.0",
) -> None:
    """Generate gt database and *_dbinfos_train.pkl for VoD/TJ4D.

    Args:
        root_path (str): Dataset root path.
        info_prefix (str): Prefix of output file names.
        out_dir (str): Directory that stores the generated info pkl files.
        version (str): Keep compatibility with original flow ('mask' toggles mask).
    """
    create_groundtruth_database(
        "KittiDataset",
        root_path,
        info_prefix,
        f"{out_dir}/{info_prefix}_infos_train.pkl",
        relative_path=False,
        mask_anno_path="instances_train.json",
        with_mask=(version == "mask"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VoD/TJ4D pkl files")
    parser.add_argument("dataset", choices=["vod", "TJ4DRadSet_LiDAR", "TJ4DRadSet_4DRadar", "TJ4DRadSet_4DRadar_filter", "TJ4DRadSet_4DRadar_image"], help="Dataset type for bookkeeping")
    parser.add_argument("--root-path", required=True, help="Dataset root path")
    parser.add_argument("--extra-tag", required=True, help="Prefix of generated pkl")
    parser.add_argument("--out-dir", default=None, help="Output dir for info/dbinfo files")
    parser.add_argument("--version", default="v1.0", help="Version string (use 'mask' if needed)")
    parser.add_argument(
        "--with-gt-db",
        action="store_true",
        help="Also generate *_dbinfos_train.pkl and gt database",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.root_path

    # VoD and TJ4D variants all share the same TJ4D/KITTI-style converter path.
    generate_vod_tj4d_infos(args.root_path, args.extra_tag)

    if args.with_gt_db:
        info_train = osp.join(out_dir, f"{args.extra_tag}_infos_train.pkl")
        if not osp.exists(info_train):
            raise FileNotFoundError(f"Cannot find train info pkl: {info_train}")
        generate_vod_tj4d_gt_database(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=out_dir,
            version=args.version,
        )


if __name__ == "__main__":
    main()
