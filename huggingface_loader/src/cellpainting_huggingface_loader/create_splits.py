import os
import json

import cv2
import pandas as pd
import numpy as np

from datasets import load_from_disk, concatenate_datasets, DatasetDict
from typing import Dict, List


class CreateSplits():
    def __init__(self, metadata_path: str = None, blob_mount_point: str = None) -> None:
        assert blob_mount_point, "Please specify blob_mount_point."

        self.metadata_path = metadata_path
        self._metadata_df = None
        self.blob_mount_point = blob_mount_point

    @property
    def metadata_df(self):
        """Lazy-load metadata parquet only when needed (not for by_plate splits)."""
        if self._metadata_df is None:
            assert self.metadata_path, "metadata_path required for source/batch splits."
            self._metadata_df = pd.read_parquet(
                f"{self.metadata_path}/plate_metadata/metadata.parquet"
            )
        return self._metadata_df
    
    def generate_dataset_splits(self, 
                                split_type: str, 
                                source_split: List[str] = None, 
                                train_size: float = 0.8, 
                                val_size: float = 0.1, 
                                test_size: float = 0.1,
                                verbose: bool = False):
        # Load source, batch, and plate info
        source_data, batch_data, plate_data = self._load_source_batch_plate_info()

        # Specifying sources that will be included in the dataset
        total_images, source_split = self._specify_split(source_data, source_split)
                
        # Using the list make list of source/batch/plate names and their image counts
        name_count_list = self._generate_name_count_list(source_split, source_data, batch_data, plate_data, split_type)

        # Use this list in the greedy algorithm to decide the splits
        train, val, test = self._greedy_find_splits(name_count_list, total_images, train_size, val_size, test_size)

        # Count the image count in each split and give a warning if the split is off more than 0.1
        actual_train_size, actual_val_size, actual_test_size = self._count_split_image_count(total_images, train, val, test)

        if abs(actual_train_size - train_size) > 0.1 or abs(actual_val_size - val_size) > 0.1 or abs(actual_test_size - test_size) > 0.1:
            print("Warning: Generated splits have a difference of more than 0.1 from the desired split size. Set verbose=True to see the splits.")

        # Generate the huggingface dataset
        train, val, test = [name for name, _ in train], [name for name, _ in val], [name for name, _ in test]
        huggingface_dataset = self._create_hf_dataset(split_type, train, val, test)
        return (train, val, test), huggingface_dataset 

    def create_downsampled_dataset(self, splits, split_type, downsample_size: float = 0.2):
        train, val, test = splits
        source_data, batch_data, plate_data = self._load_source_batch_plate_info()

        downsampled_train = self._downsample_data(train, source_data, batch_data, downsample_size, split_type)
        downsampled_val = self._downsample_data(val, source_data, batch_data, downsample_size, split_type)
        downsampled_test = self._downsample_data(test, source_data, batch_data, downsample_size, split_type)
        
        if split_type == "by_source":
            downsampled_dataset = self._create_hf_dataset("by_batch", downsampled_train, downsampled_val, downsampled_test)
        elif split_type == "by_batch":
            downsampled_dataset = self._create_hf_dataset("by_plate", downsampled_train, downsampled_val, downsampled_test)
        else:
            raise ValueError("Invalid split type.")
    
        return (downsampled_train, downsampled_val, downsampled_test), downsampled_dataset

    def filter_dataset(self, dataset, JCP_list):
        def filter_JCP_metadata(example):
            filename = example["filename"]
            JCP_metadata = self.metadata_df[self.metadata_df["agp_path"] == filename]["Metadata_JCP2022"]
            return JCP_metadata not in JCP_list
        return dataset.filter(filter_JCP_metadata)

    def _downsample_data(self, sources, source_data, batch_data, downsample_size, split_type):
        downsampled_data = []
        for name in sources:
            if split_type == "by_source":
                batches = source_data[name]["batches"]
            elif split_type == "by_batch":
                batches = batch_data[name]["plates"]
            else:
                raise ValueError("Invalid split_type")
            
            downsample_index = int(len(batches) * downsample_size)
            downsampled_data.extend(batches[:downsample_index])
        return downsampled_data
        
    def _load_source_batch_plate_info(self):
        with open(f'{self.metadata_path}/source_batch_plate_info/source_info.json', 'r') as file:
            source_data = json.load(file)
        with open(f'{self.metadata_path}/source_batch_plate_info/batch_info.json', 'r') as file:
            batch_data = json.load(file)
        with open(f'{self.metadata_path}/source_batch_plate_info/plate_info.json', 'r') as file:
            plate_data = json.load(file)

        return source_data, batch_data, plate_data

    def _specify_split(self, source_data, source_split):
        if not source_split:
            source_split = source_data.keys()
        total_images = sum([source_data[source]["num_images"] for source in source_split])

        return total_images, source_split

    def _calculate_count_splits(self, total_images, train_size, val_size, test_size):
        train_set_ct, val_set_ct, test_set_ct = total_images*train_size, total_images*val_size, total_images*test_size
        return train_set_ct, val_set_ct, test_set_ct

    def _generate_name_count_list(self, source_split, source_data, batch_data, plate_data, split_type):
        batch_names = []
        for source in source_split:
            batch_names.extend(source_data[source]["batches"])

        plate_names = []
        for batch in batch_names:
            plate_names.extend(batch_data[batch]["plates"])

        name_count = []
        match split_type:
            case "by_source":
                name_count = [(source, source_data[source]["num_images"]) for source in source_split]
                name_count = sorted(name_count, key=lambda x: x[1], reverse=True)
            case "by_batch":
                name_count = [(batch, batch_data[batch]["num_images"]) for batch in batch_names]
                name_count = sorted(name_count, key=lambda x: x[1], reverse=True)
            case "by_plate":
                name_count = [(plate, plate_data[plate]["num_images"]) for plate in plate_names]
            case _:
                raise ValueError("Split type is not valid. Valid split types are 'by_source', 'by_batch', and 'by_plate'.")
        return name_count

    def _greedy_find_splits(self, name_count, total_images, train_size, val_size, test_size):
        train, val, test = [], [], []
        train_ct, val_ct, test_ct = 0, 0, 0
        train_set_ct, val_set_ct, test_set_ct = self._calculate_count_splits(total_images, train_size, val_size, test_size)

        for name, img_count in name_count:
            if test_ct + img_count <= test_set_ct or (val_ct >= val_set_ct and test_ct >= test_set_ct):
                test.append((name, img_count))
                test_ct += img_count
            elif val_ct + img_count <= val_set_ct or (test_ct >= test_set_ct):
                val.append((name, img_count))
                val_ct += img_count
            else:
                train.append((name, img_count))
                train_ct += img_count

        if not val and not test:
            if val_size > train_size:
                test.append(train.pop())
                val.append(train.pop())
            else:
                val.append(train.pop())
                test.append(train.pop())

        if not val:
            val.append(train.pop())
        if not test:
            test.append(train.pop())
        
        return train, val, test

    def _count_split_image_count(self, total_images, train, val, test):
        actual_train_size = sum(count[1] for count in train)/total_images
        actual_val_size = sum(count[1] for count in val)/total_images
        actual_test_size = sum(count[1] for count in test)/total_images

        return actual_train_size, actual_val_size, actual_test_size

    def _create_hf_dataset(self, split_type, train, val, test):
        split_functions = {
            "by_source": self._split_source,
            "by_batch": self._split_batch,
            "by_plate": self._split_plate
        }
        
        if split_type not in split_functions:
            raise ValueError("Split type is not valid. Valid split types are 'by_source', 'by_batch', and 'by_plate'. Terminating.")
        
        split_function = split_functions[split_type]
        
        return DatasetDict({
            "train": split_function(train),
            "val": split_function(val),
            "test": split_function(test)
        })

    def _split_source(self, split):
        plate_list = self.metadata_df[self.metadata_df["Metadata_Source"].isin(split)]["Metadata_Plate"].unique()
        dataset = concatenate_datasets([load_from_disk(f"{self.blob_mount_point}/{plate_name}")["train"] for plate_name in plate_list])
        return dataset
    def _split_batch(self, split):
        plate_list = self.metadata_df[self.metadata_df["Metadata_Batch"].isin(split)]["Metadata_Plate"].unique()
        dataset = concatenate_datasets([load_from_disk(f"{self.blob_mount_point}/{plate_name}")["train"] for plate_name in plate_list])
        return dataset
    def _split_plate(self, split):
        plate_list = split
        dataset = concatenate_datasets([load_from_disk(f"{self.blob_mount_point}/{plate_name}")["train"] for plate_name in plate_list])
        return dataset

    def generate_dataset_by_name(self, split_type: str, split: List[str]):
        split_functions = {
            "by_source": self._split_source,
            "by_batch": self._split_batch,
            "by_plate": self._split_plate
        }
        if split_type not in split_functions:
            raise ValueError("Split type is not valid. Valid split types are 'by_source', 'by_batch', and 'by_plate'. Terminating.")
        
        split_function = split_functions[split_type]
        return split_function(split)
        