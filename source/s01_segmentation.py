import pandas as pd

from cellpose import models
from faim_ipa.utils import create_logger
from faim_ipa.histogram import UIntHistogram
from metamorph_mda_parser.nd import NdInfo
from pathlib import Path
from skimage.measure import regionprops_table
from tifffile.tifffile import imwrite

from config import SegmentAggresomesConfig, SegmentNucleiConfig


def segment(array, model: models.CellposeModel):
    mask, _, _ = model.eval(array, diameter=80)
    return mask


def process_nd_file(
    nd_file: Path,
    output_folder: Path,
    nuc_channel_index: int,
    ub_channel_index: int,
):
    print(f"Processing {nd_file.name}.")
    ndinfo = NdInfo.from_path(nd_file)
    # files = ndinfo.get_files()
    data_array = ndinfo.get_data_array(channels=[nuc_channel_index])
    image = data_array.data.compute().squeeze()

    model = models.CellposeModel(gpu=True, pretrained_model="nuclei")
    # labels = segment(data_array.sel(channel=0).data.compute(), model=model)
    labels = segment(image, model=model)

    raw_output_dir = output_folder / "raw"
    raw_output_dir.mkdir(exist_ok=True, parents=True)
    ub_output_dir = output_folder / "ub"
    ub_output_dir.mkdir(exist_ok=True, parents=True)
    segmentation_output_dir = output_folder / "segmentation"
    segmentation_output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Data shape: {image.shape} {image.dtype}")
    print(f"Labels shape: {labels.shape} {labels.dtype}")

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

    segmentation_path = segmentation_output_dir / (nd_file.with_suffix(".tif").name)
    imwrite(
        segmentation_path,
        labels,
        imagej=True,
    )

    table = regionprops_table(
        label_image=labels,
        properties=("label", "centroid"),
    )
    table_df = pd.DataFrame(table)
    table_path = segmentation_output_dir / (nd_file.with_suffix(".csv").name)
    table_df.to_csv(
        table_path,
        index=False,
    )

    print("Done")
    return [
        {
            "uri": raw_path.relative_to(output_folder.parent).as_posix(),
            "name": nd_file.stem,
            "type": "image",
            "channel": 0,
            "view": "default",
            "grid": "dapi",
            "labels_table": "",
        },
        {
            "uri": ub_path.relative_to(output_folder.parent).as_posix(),
            "name": nd_file.stem,
            "type": "image",
            "channel": 0,
            "view": "default",
            "grid": "ub",
            "labels_table": "",
        },
        {
            "uri": segmentation_path.relative_to(output_folder.parent).as_posix(),
            "name": nd_file.name,
            "type": "labels",
            "channel": 0,
            "view": "default",
            "grid": "nuclei",
            "labels_table": table_path.relative_to(output_folder.parent).as_posix(),
        },
    ]


def run(config: SegmentNucleiConfig):
    logger = create_logger("segment_nuclei")

    # list all nd files in all subfolders
    subfolders = [
        f
        for f in config.parent_folder.iterdir()
        if f.is_dir() and not f.name == "Montage"
    ]

    image_entries = []
    image_names = []

    for folder in subfolders:
        nds = [nd for nd in folder.glob("*.nd")]
        output_folder = config.output_folder / folder.name

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
                    )
                )
                image_names.append(nd.stem)
            except RuntimeError as e:
                logger.error(f"Error processing {nd.name}: {e}")

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
