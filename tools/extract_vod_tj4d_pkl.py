"""Extracted VoD/TJ4D dataset pkl generation pipeline.

This file centralizes the pkl-related generation logic that is otherwise
scattered across `tools/create_data.py`, `tools/data_converter/tj4d_converter.py`
and `tools/data_converter/create_gt_database.py`.
"""

import argparse

from tools.data_converter import tj4d_converter as tj4d


def _dataset_converter_options(dataset: str) -> dict:
    if dataset == "vod":
        return dict(img_file_tail=".jpg", use_prefix_id=False, num_features=7)
    return dict(img_file_tail=".png", use_prefix_id=True, num_features=8)


def generate_vod_tj4d_infos(root_path: str, info_prefix: str, dataset: str) -> None:
    """Generate *_infos_{train|val|trainval|test}.pkl and reduced point cloud.

    Args:
        root_path (str): Dataset root path.
        info_prefix (str): Prefix of output pkl names.
    """
    # Same core flow as `kitti_data_prep` in tools/create_data.py for VoD/TJ4D.
    converter_options = _dataset_converter_options(dataset)
    tj4d.create_kitti_info_file(root_path, info_prefix, **converter_options)
    tj4d.create_reduced_point_cloud(
        root_path, info_prefix, num_features=converter_options["num_features"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VoD/TJ4D pkl files")
    parser.add_argument("dataset", choices=["vod", "TJ4DRadSet_LiDAR", "TJ4DRadSet_4DRadar", "TJ4DRadSet_4DRadar_filter", "TJ4DRadSet_4DRadar_image"], help="Dataset type for bookkeeping")
    parser.add_argument("--root-path", required=True, help="Dataset root path")
    parser.add_argument("--extra-tag", required=True, help="Prefix of generated pkl")
    parser.add_argument(
        "--with-gt-db",
        action="store_true",
        help="Unsupported for the current VoD/TJ4D RCDFNet configs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.with_gt_db:
        raise NotImplementedError(
            "--with-gt-db is disabled for this standalone VoD/TJ4D flow. "
            "The provided RCDFNet configs do not use database sampling, and "
            "the old GT database path is not dataset-aware.")

    # VoD and TJ4D variants share the KITTI-style converter path, with
    # dataset-specific file suffix, id width, and radar point dimensions.
    generate_vod_tj4d_infos(args.root_path, args.extra_tag, args.dataset)


if __name__ == "__main__":
    main()
