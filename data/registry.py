"""Dataset registry for the four OCSFlow benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    image_file: str
    gt_file: str
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


SPECS = {
    "IP": DatasetSpec("Indian_pines_corrected.mat", "Indian_pines_gt.mat", (
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
        "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
        "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
        "Buildings-grass-trees-drives", "Stone-steel-towers")),
    "PU": DatasetSpec("PaviaU.mat", "PaviaU_gt.mat", (
        "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
        "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows")),
    "KSC": DatasetSpec("KSC.mat", "KSC_gt.mat", (
        "Scrub", "Willow swamp", "Cabbage palm hammock", "Cabbage palm/oak hammock",
        "Slash pine", "Oak/broadleaf hammock", "Hardwood swamp", "Graminoid marsh",
        "Spartina marsh", "Cattail marsh", "Salt marsh", "Mud flats", "Water")),
    "HC": DatasetSpec("WHU_Hi_HanChuan.mat", "WHU_Hi_HanChuan_gt.mat", (
        "Strawberry", "Cowpea", "Soybean", "Sorghum", "Water spinach", "Watermelon",
        "Greens", "Trees", "Grass", "Red roof", "Gray roof", "Plastic", "Bare soil",
        "Road", "Bright object", "Water")),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    code = name.strip().upper()
    if code not in SPECS:
        raise KeyError(f"unsupported dataset {name!r}; choose IP, PU, KSC or HC")
    return SPECS[code]
