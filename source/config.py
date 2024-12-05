from faim_ipa.utils import IPAConfig, prompt_with_questionary
from pydantic import DirectoryPath, Field
from typing import Annotated


class SegmentNucleiConfig(IPAConfig):
    parent_folder: DirectoryPath = Field(
        ..., description="Parent folder for all nd files to be processed"
    )
    output_folder: DirectoryPath = Field(..., description="Output folder")
    nucleus_channel_index: Annotated[int, Field(strict=True, ge=0)]
    ubiquitin_channel_index: Annotated[int, Field(strict=True, ge=0)]

    @staticmethod
    def config_name():
        return "s01_config.yaml"


class SegmentAggresomesConfig(IPAConfig):
    results_folder: DirectoryPath = Field(
        ..., description="Folder containing the results of the segmentation"
    )
    image_names: list[str] = Field(..., description="List of image names to process")

    @staticmethod
    def config_name():
        return "s02_config.yaml"


if __name__ == "__main__":
    try:
        defaults = SegmentNucleiConfig.load().model_dump()
    except FileNotFoundError:
        defaults = {}

    result = prompt_with_questionary(SegmentNucleiConfig, defaults=defaults)

    result.save()
