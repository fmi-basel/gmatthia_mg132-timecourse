from faim_ipa.utils import IPAConfig, prompt_with_questionary
from pydantic import DirectoryPath, Field
from typing import Annotated


class SegmentNucleiConfig(IPAConfig):
    parent_folder: DirectoryPath = "raw_data"
    output_folder: DirectoryPath = "processed_data"
    nucleus_channel_index: Annotated[int, Field(strict=True, ge=0)] = 0
    diameter: int = 150
    flow_threshold: float = 0.8
    cellprob_threshold: float = 0.0
    ubiquitin_channel_index: Annotated[int, Field(strict=True, ge=0)] = 2
    threshold_factor: float = 3.0

    @staticmethod
    def config_name():
        return "s01_config.yaml"


class SegmentAggresomesConfig(IPAConfig):
    results_folder: DirectoryPath = Field(
        ..., description="Folder containing the results of the segmentation"
    )
    image_names: dict[str, list[str]] = Field(
        ..., description="List of image names to process"
    )
    threshold_factor: float = 3.0

    @staticmethod
    def config_name():
        return "s02_config.yaml"


if __name__ == "__main__":
    try:
        defaults = SegmentNucleiConfig.load().model_dump()
    except FileNotFoundError:
        defaults = SegmentNucleiConfig().model_dump()

    result = prompt_with_questionary(SegmentNucleiConfig, defaults=defaults)

    result.save()
