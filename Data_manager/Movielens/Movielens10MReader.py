#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile, shutil
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.Movielens._utils_movielens_parser import _loadICM_tags, _loadICM_genres, _loadURM

def _loadURM(URM_path, header=None, separator=','):
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep=separator, header=header, dtype={
                                    0: int, 1: int, 2: float})

    URM_all_dataframe.columns = ["user_id", "item_id", "rating"]

    return URM_all_dataframe


def _loadICM(ICMpath, header=None, separator=','):
    ICM_all_dataframe = pd.read_csv(filepath_or_buffer=ICMpath, sep=separator, header=header, dtype={
                                    0: int, 1: int, 2: float})

    ICM_all_dataframe.columns = ["item_id", "attribute_id", "value"]

    return ICM_all_dataframe


class Movielens10MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    DATASET_SUBFOLDER = "Movielens10M/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_tags"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        URM_path = '../../data/data_train.csv'
        ICM_path = '../../data/data_ICM_title_abstract.csv'


        self._print("Loading Interactions")
        URM_all_dataframe = _loadURM(URM_path, header=None, separator=',')
        ICM_all_dataframe = _loadICM(ICM_path, header=None, separator=',')

        # ICM_all_dataframe = pd.concat([ICM_genres_dataframe, ICM_tags_dataframe])

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")


        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),is_implicit=self.IS_IMPLICIT)

        return loaded_dataset
