import argparse
from typing import Any, List, Optional

import numpy as np
import scipy.io

IMAGE_PATTERN_ID: int = {"easy": 17, "normal": 102}
IMAGE_TYPE: str = {0: "imagelabels", 1:"set"}
INPUT_DIR_TMP: str = "../data/oxfordflower{}/{}.mat"
OUTPUT_DIR_TMP: str = "../data/oxfordflower{}/{}_family.mat"

init_id: str = "easy"
init_type: int = 0


label_to_name: dict[int, Optional[dict[int, str]]] = {
    # 頑張ってて入力します
    # 参考; https://aidiary.hatenablog.com/entry/20170131/1485864665
    17: {
        1: "Tulip",
        2: "Snowdrop",
        3: "LilyValley",
        4: "Bluebell",
        5: "Crocus",
        6: "Iris",
        7: "Tigerlily",
        8: "Daffodil",
        9: "Fritillary",
        10: "Sunflower",
        11: "Daisy",
        12: "ColtsFoot",
        13: "Dandelion",
        14: "Cowslip",
        15: "Buttercup",
        16: "Windflower",
        17: "Pansy"
    },
    102: None
}

family_to_name: dict[int, Optional[dict[str, List[str]]]] = {
    17: {
        "Liliaceae": [
            "Tulip",
            "LilyValley",
            "Tigerlily",
            "Fritillary"
        ],
        "Amaryllidaceae": ["Snowdrop", "Daffodil"],
        "Asparagaceae": ["Bluebell", ],
        "Iridaceae": ["Crocus", "Iris"],
        "Asteraceae": [
            "Sunflower",
            "Daisy",
            "ColtsFoot",
            "Dandelion"
        ],
        "Primulaceae": ["Cowslip"],
        "Ranunculaceae": ["Buttercup", "Windflower"],
        "Violaceae": ["Pansy"],
    },
    102: [None]

}


def make_labels(names: Optional[List[str]] = None) -> dict[int, str]:
    if None in names:
        return
    output: dict[int, str] = {}
    for i, name in enumerate(names):
        output[i + 1] = name
    return output


family_label: dict[int, Optional[dict[int, str]]] = {
    17: make_labels(list(family_to_name[17])),
    102: make_labels(list(family_to_name[102])),
}


def get_family_label(idx: str,
                     label_to_name: dict[int, str],
                     family_to_name: dict[str, List[str]],
                     family_label: dict[int, str]):
    for _idx, name in label_to_name.items():
        if idx == _idx:
            break

    for family, labels in family_to_name.items():
        if name in labels:
            break

    for idx, _family in family_label.items():
        if family == _family:
            return idx


def get_error_text(label_to_name: Optional[dict[int, str]] = None,
                   family_to_name: Optional[dict[str, List[str]]] = None,
                   family_label: Optional[dict[int, str]] = None) -> str:
    txt: str = ""
    if label_to_name is None:
        txt += "You must set a mapping dictionary between label number and a flower name before.\n"
    if family_to_name is None:
        txt += "You must set a mapping dictionary between family of flowers and flowers before.\n"
    if family_label is None:
        txt += (
            "You must set a mapping dictionary between label number and family of flowers before. "
            "If you have already set family_to_name, what you need to do is just setting a mapping dictionary."
        )
    return txt


def main(args):
    image_size: int = {"easy": 17, "normal": 102}[args.size]
    image_type: str = {0: "imagelabels", 1:"set"}[args.type]
    filename: str = INPUT_DIR_TMP.format(image_size, image_type)

    image_dict: dict[str, Any] = scipy.io.loadmat(filename)

    if "labels" in image_dict:
        family_labels = np.zeros_like(image_dict["labels"], dtype=np.uint8)
        l2n = label_to_name[image_size]
        f2n = family_to_name[image_size]
        f_label = family_label[image_size]

        txt: str = get_error_text(l2n, f2n, f_label)
        if len(txt) > 0:
            raise ValueError(txt)
        labels = image_dict["labels"][0]

        for i, label in enumerate(labels):
            family_labels[0, i] = get_family_label(label, l2n, f2n, f_label)

        image_dict["family_labels"] = family_labels
        filename_family: str = OUTPUT_DIR_TMP.format(image_size, image_type)
        scipy.io.savemat(filename_family, image_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default="easy")
    parser.add_argument('--type', type=int, default=0)
    args = parser.parse_args()

    main(args)
