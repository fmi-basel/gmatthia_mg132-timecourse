import numpy as np
import pandas as pd

from faim_ipa.histogram import UIntHistogram
from faim_ipa.utils import create_logger
from pathlib import Path
from rich.pretty import pretty_repr
from skimage.measure import label, regionprops_table
from tifffile import imread, imwrite

from config import SegmentAggresomesConfig


def _get_histogram(results_folder: Path, image_names: list[str]):
    joined_histogram = UIntHistogram()
    for subfolder in image_names:
        for image_name in image_names[subfolder]:
            ub_hist_path = (results_folder / subfolder / "ub" / image_name).with_suffix(
                ".npz"
            )
            ub_hist = UIntHistogram.load(ub_hist_path)
            joined_histogram.combine(ub_hist)
    return joined_histogram


def _segment_aggresomes(
    image_path: Path,
    aggresome_threshold: float,
    aggresome_folder: str,
    nuclei_folder: str,
):
    # load ub image and threshold with aggresome threshold
    ub_image = imread(image_path)
    aggresomes = ub_image > aggresome_threshold
    nuc_image = imread(nuclei_folder / image_path.name)
    # mask aggresomes with nuclei
    aggresomes = aggresomes & (nuc_image == 0)
    aggresome_labels = label(aggresomes)

    # save aggresome mask
    aggresome_path = aggresome_folder / image_path.name
    imwrite(aggresome_path, aggresome_labels.astype(np.uint8))
    return aggresome_labels, aggresome_path


def main(config: SegmentAggresomesConfig):
    logger = create_logger("segment_aggresomes")
    logger.info(pretty_repr(config))

    joined_histogram = _get_histogram(config.results_folder, config.image_names)

    aggresome_threshold = (
        joined_histogram.mean() + config.threshold_factor * joined_histogram.std()
    )
    logger.info(f"Using aggresome threshold: {aggresome_threshold}")

    entries = []
    for subfolder in config.image_names:
        logger.info(f"Processing subfolder {subfolder}...")
        aggresome_dir = config.results_folder / subfolder / "aggresome"
        aggresome_dir.mkdir(parents=True, exist_ok=True)
        nuclei_dir = config.results_folder / subfolder / "nuclei"

        for image_name in config.image_names[subfolder]:
            logger.info(f"\tProcessing {image_name}...")
            image_path = (
                config.results_folder / subfolder / "ub" / image_name
            ).with_suffix(".tif")
            aggresome_labels, aggresome_path = _segment_aggresomes(
                image_path,
                aggresome_threshold,
                aggresome_dir,
                nuclei_dir,
            )

            entries.append(
                {
                    "uri": aggresome_path.relative_to(config.results_folder).as_posix(),
                    "name": aggresome_path.stem,
                    "type": "labels",
                    "channel": 0,
                    "view": "default",
                    "grid": "aggresomes",
                    "labels_table": "",
                }
            )

            # count and measure aggresomes per cell
            cells_path = (
                config.results_folder / subfolder / "cells" / image_name
            ).with_suffix(".tif")
            cells_image = imread(cells_path)

            intensities = regionprops_table(
                label_image=cells_image,
                intensity_image=(aggresome_labels > 0),
                properties=("label", "intensity_mean", "num_pixels"),
            )
            intensities_table = pd.DataFrame(intensities)
            intensities_table["aggresome_size"] = (
                intensities_table["num_pixels"] * intensities_table["intensity_mean"]
            )

            # count number of unique aggresomes per cell
            cell_id_table = regionprops_table(
                label_image=aggresome_labels,
                intensity_image=cells_image,
                properties=("label", "intensity_max"),
            )
            cell_id_table = pd.DataFrame(cell_id_table)
            counts_table = cell_id_table.groupby("intensity_max")["label"].count()

            cells_table_path = (
                config.results_folder / subfolder / "cells" / image_name
            ).with_suffix(".csv")
            cells_table = pd.read_csv(cells_table_path)

            merged_table = cells_table.merge(
                intensities_table, on="label", how="left"
            ).merge(
                counts_table,
                left_on="label",
                right_on="intensity_max",
                suffixes=["", "_count"],
                how="left",
            )
            # set absent values to 0
            merged_table["label_count"] = merged_table["label_count"].fillna(0)

            merged_table.to_csv(
                cells_table_path,
                index=False,  # columns=["label", "centroid-0", "centroid-1", "aggresome_size"]
            )

    df = pd.DataFrame(entries)

    collection_table_path = config.results_folder / "mobie_collection.tsv"
    collection_table = pd.read_csv(collection_table_path, sep="\t")

    pd.concat([collection_table, df], ignore_index=True).to_csv(
        collection_table_path, sep="\t", index=False
    )

    logger.info("Done.")


if __name__ == "__main__":
    config = SegmentAggresomesConfig.load()
    main(config)
