import argparse
import json
import os

from tqdm import tqdm

from goodreads_data import GoodreadsData
from lastfm_data import LastfmData
from steam_data import SteamData


DATASET_REGISTRY = {
    "goodreads": GoodreadsData,
    "lastfm": LastfmData,
    "steam": SteamData,
}


def export_dataset(dataset_name: str, data_dir: str, output_dir: str, cans_num: int = 20):
    dataset_cls = DATASET_REGISTRY[dataset_name]
    splits = ["train", "val", "test"]
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        dataset = dataset_cls(data_dir=data_dir, stage=split, cans_num=cans_num)
        records = []
        for i in tqdm(range(len(dataset)), desc=f"{dataset_name}-{split}"):
            sample = dataset[i]
            record = {
                "historyList": sample["movie_seq"],
                "itemList": sample["cans_name"],
                "trueSelection": sample["next_title"],
            }
            records.append(record)

        output_path = os.path.join(output_dir, f"{dataset_name}-{split}.json")
        with open(output_path, "w") as f:
            json.dump(records, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT-style JSON data from reference datasets.")
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY.keys()), required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cans_num", type=int, default=20)
    args = parser.parse_args()

    export_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cans_num=args.cans_num,
    )
