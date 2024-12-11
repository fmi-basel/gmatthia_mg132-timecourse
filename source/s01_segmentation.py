import pandas as pd

from cellpose import models
from faim_ipa.utils import create_logger
from faim_ipa.histogram import UIntHistogram
from metamorph_mda_parser.nd import NdInfo
from pathlib import Path
from rich.pretty import pretty_repr
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from tifffile.tifffile import imwrite

from config import SegmentAggresomesConfig, SegmentNucleiConfig


def segment(
    array,
    model: models.CellposeModel,
    diameter: int,
    flow_threshold: float,
    cellprob_threshold: float,
):
    mask, _, _ = model.eval(
        array,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return mask


def process_nd_file(
    nd_file: Path,
    output_folder: Path,
    nuc_channel_index: int,
    ub_channel_index: int,
    model: models.CellposeModel,
    diameter: int,
    flow_threshold: float,
    cellprob_threshold: float,
    logger,
):
    logger.info(f"Processing {nd_file.name}...")
    ndinfo = NdInfo.from_path(nd_file)
    # files = ndinfo.get_files()
    data_array = ndinfo.get_data_array(channels=[nuc_channel_index])
    image = data_array.data.compute().squeeze()
    logger.info(f"{image.shape=} {image.dtype=}")

    logger.info("Segmenting nuclei...")
    nuclei = segment(
        image,
        model=model,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    logger.info("Expanding nuclei labels...")
    cells = expand_labels(nuclei, distance=50)

    raw_output_dir = output_folder / "raw"
    raw_output_dir.mkdir(exist_ok=True, parents=True)
    ub_output_dir = output_folder / "ub"
    ub_output_dir.mkdir(exist_ok=True, parents=True)
    nuclei_output_dir = output_folder / "nuclei"
    nuclei_output_dir.mkdir(exist_ok=True, parents=True)
    cells_output_dir = output_folder / "cells"
    cells_output_dir.mkdir(exist_ok=True, parents=True)

    # save Ub channel
    ub_array = ndinfo.get_data_array(channels=[ub_channel_index])
    ub_image = ub_array.data.compute().squeeze()

    # save Ub histogram
    ub_hist = UIntHistogram(ub_image)
    ub_hist_path = ub_output_dir / (nd_file.with_suffix(".npz").name)
    ub_hist.save(ub_hist_path)

    raw_path = raw_output_dir / (nd_file.with_suffix(".tif").name)
    imwrite(raw_path, image, imagej=True)

    ub_path = ub_output_dir / (nd_file.with_suffix(".tif").name)
    imwrite(ub_path, ub_image, imagej=True)

    nuclei_path = nuclei_output_dir / (nd_file.with_suffix(".tif").name)
    imwrite(
        nuclei_path,
        nuclei,
        imagej=True,
    )

    cells_path = cells_output_dir / (nd_file.with_suffix(".tif").name)
    imwrite(
        cells_path,
        cells,
        imagej=True,
    )

    table = regionprops_table(
        label_image=nuclei,
        properties=("label", "centroid"),
    )
    table_df = pd.DataFrame(table)
    table_path = nuclei_output_dir / (nd_file.with_suffix(".csv").name)
    table_df.to_csv(
        table_path,
        index=False,
    )

    table_cells = regionprops_table(
        label_image=cells,
        properties=("label", "centroid"),
    )
    table_cells_df = pd.DataFrame(table_cells)
    table_cells_path = cells_output_dir / (nd_file.with_suffix(".csv").name)
    table_cells_df.to_csv(
        table_cells_path,
        index=False,
    )

    logger.info("Done.")
    return [
        {
            "uri": raw_path.relative_to(output_folder.parent).as_posix(),
            "name": nd_file.stem + "_dapi",
            "type": "intensities",
            "channel": 0,
            "view": "default",
            "grid": "dapi",
            "labels_table": "",
        },
        {
            "uri": ub_path.relative_to(output_folder.parent).as_posix(),
            "name": nd_file.stem + "_ub",
            "type": "intensities",
            "channel": 0,
            "view": "default",
            "grid": "ub",
            "labels_table": "",
        },
        {
            "uri": nuclei_path.relative_to(output_folder.parent).as_posix(),
            "name": nd_file.stem + "_nuclei",
            "type": "labels",
            "channel": 0,
            "view": "default",
            "grid": "nuclei",
            "labels_table": table_path.relative_to(output_folder.parent).as_posix(),
        },
        {
            "uri": cells_path.relative_to(output_folder.parent).as_posix(),
            "name": nd_file.stem + "_cells",
            "type": "labels",
            "channel": 0,
            "view": "default",
            "grid": "cells",
            "labels_table": table_cells_path.relative_to(
                output_folder.parent
            ).as_posix(),
        },
    ]


def run(config: SegmentNucleiConfig):
    logger = create_logger("segment_nuclei")
    logger.info(pretty_repr(config))

    # list all nd files in all subfolders
    subfolders = [
        f
        for f in config.parent_folder.iterdir()
        if f.is_dir() and not f.name == "Montage"
    ]

    image_entries = []
    image_names = {}

    model = models.CellposeModel(gpu=True, pretrained_model="nuclei")

    for folder in subfolders:
        nds = [nd for nd in folder.glob("*.nd")]
        output_folder = config.output_folder / folder.name

        image_list = []
        for nd in nds:
            # process each nd file
            logger.info(f"Processing {nd.name}")
            try:
                image_entries.extend(
                    process_nd_file(
                        nd_file=nd,
                        output_folder=output_folder,
                        nuc_channel_index=config.nucleus_channel_index,
                        ub_channel_index=config.ubiquitin_channel_index,
                        model=model,
                        diameter=config.diameter,
                        flow_threshold=config.flow_threshold,
                        cellprob_threshold=config.cellprob_threshold,
                        logger=logger,
                    )
                )
                image_list.append(nd.stem)
            except RuntimeError as e:
                logger.error(f"Error processing {nd.name}: {e}")

        image_names[folder.name] = image_list

    # write mobie collection table
    collection_table = pd.DataFrame(image_entries)
    collection_table.to_csv(
        config.output_folder / "mobie_collection.tsv", sep="\t", index=False
    )

    # write config for next step
    next_config = SegmentAggresomesConfig(
        results_folder=config.output_folder,
        image_names=image_names,
    )
    next_config.save()


if __name__ == "__main__":
    config = SegmentNucleiConfig.load()
    run(config=config)
