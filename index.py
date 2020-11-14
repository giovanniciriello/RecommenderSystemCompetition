import os
from typing import Tuple, Callable, Dict, Optional, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split

def load_data(file_name):
	return pd.read_csv("./data/{}.csv".format(file_name),
		names=["user_id", "item_id", "rating"],
		header=0,
		dtype={
			"user_id": np.uint32,
			"item_id": np.uint32,
			"rating": np.bool_ #np.double
		})

def preprocess_data(ratings: pd.DataFrame):
	unique_users = ratings.user_id.unique()
	unique_items = ratings.item_id.unique()

	num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
	num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

	mapping_user_id = pd.DataFrame({
		"mapped_user_id" : np.arange(num_users),
		"user_id" : unique_users
	})

	mapping_item_id = pd.DataFrame({
		"mapped_item_id" : np.arange(num_items),
		"item_id" : unique_items
	})

	ratings = pd.merge(left=ratings, right=mapping_user_id, how="inner", on="user_id")
	ratings = pd.merge(left=ratings, right=mapping_item_id, how="inner", on="item_id")

	return ratings, num_users, num_items

def dataset_splits(ratings, num_users, num_items, validation_percentage: float, testing_percentage: float):
	seed = 1516

	(user_ids_training, user_ids_test,
	item_ids_training, item_ids_test,
	ratings_training, ratings_test) = train_test_split(
			ratings.mapped_user_id,
			ratings.mapped_item_id,
			ratings.rating,
			test_size=testing_percentage,
			shuffle=True,
			random_state=seed)

	(user_ids_training, user_ids_validation,
	item_ids_training, item_ids_validation,
	ratings_training, ratings_validation) = train_test_split(
			user_ids_training,
			item_ids_training,
			ratings_training,
			test_size=validation_percentage,
			shuffle=True,
			random_state=seed)

	urm_train = sp.csr_matrix((ratings_training, (user_ids_training, item_ids_training)),
		shape=(num_users, num_items)
	)

	urm_validation = sp.csr_matrix((ratings_validation, (user_ids_validation, item_ids_validation)),
		shape=(num_users, num_items)
	)

	urm_test = sp.csr_matrix((ratings_test, (user_ids_test, item_ids_test)),
		shape=(num_users, num_items)
	)

	return urm_train, urm_validation, urm_test


ratings = load_data("data_train")
ratings, num_users, num_items = preprocess_data(ratings)
urm_train, urm_validation, urm_test = dataset_splits(ratings, num_users, num_items, validation_percentage=0.10, testing_percentage=0.20)
