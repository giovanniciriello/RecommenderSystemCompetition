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
        "rating": np.uint32  # np.double
    })


def preprocess_data(ratings: pd.DataFrame):
    unique_users = ratings.user_id.unique()
    unique_items = ratings.item_id.unique()

    num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
    num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

    mapping_user_id = pd.DataFrame({
        "mapped_user_id": np.arange(num_users),
        "user_id": unique_users
    })

    mapping_item_id = pd.DataFrame({
        "mapped_item_id": np.arange(num_items),
        "item_id": unique_items
    })

    ratings = pd.merge(left=ratings, right=mapping_user_id,
                       how="inner", on="user_id")
    ratings = pd.merge(left=ratings, right=mapping_item_id,
                       how="inner", on="item_id")

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


def similarity(urm: sp.csc_matrix, shrink: int):
    item_weights = np.sqrt(
        np.sum(
            urm_test.tocsc().power(2),
            axis=0
        ),
    ).A.flatten()

    num_items = urm.shape[1]
    item_dot_product = urm.T.dot(urm).todense()

    weights = np.empty(shape=(num_items, num_items))
    for item_id in range(num_items):
        numerator = item_dot_product[item_id]
        denominator = item_weights[item_id] * item_weights + shrink + 1e-6

        weights[item_id] = numerator / denominator

    np.fill_diagonal(weights, 0.0)

    return weights


class CFItemKNN(object):

    def __init__(self, shrink: int):
        self.shrink = shrink
        self.weights = None

    # Callback[[sp.csc_matrix, int], np.array]
    def fit(self, urm_train: sp.csc_matrix, similarity_function):
        if not sp.isspmatrix_csc(urm_train):
            raise TypeError(
                f"We expected a CSC matrix, we got {type(urm_train)}")

        self.weights = similarity_function(urm_train, self.shrink)

    def recommend(self, user_id: int, urm_train: sp.csr_matrix, at: Optional[int] = None, remove_seen: bool = True):

        user_profile = urm_train[user_id]

        ranking = user_profile.dot(self.weights).flatten()
        if remove_seen:
            user_profile_start = urm_train.indptr[user_id]
            user_profile_end = urm_train.indptr[user_id+1]

            seen_items = urm_train.indices[user_profile_start:user_profile_end]

            ranking[seen_items] = -np.inf

        ranking = np.flip(np.argsort(ranking))

        return ranking[:at]


def recall(recommentations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.in1d(recommentations, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant) / relevant_items.shape[0]
    return recall_score


def precision(recommentations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.in1d(recommentations, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant) / recommentations.shape[0]
    return precision_score


def mean_average_precision(recommentations: np.array, relevant_items: np.array) -> float:

    is_relevant = np.in1d(recommentations, relevant_items, assume_unique=True)
    p_at_k = is_relevant * \
        np.cumsum(is_relevant, dtype=np.float32) / \
        (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / \
        np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score


def evaluator(recommender: object, urm_train: sp.csr_matrix, URM_test: sp.csr_matrix):

    num_recommendations = 10
    cumulative_precision = 0
    cumulative_recall = 0
    cumulative_MAP = 0

    num_users = urm_train.shape[0]
    num_users_evaluated = 0
    num_users_skipped = 0

    for user_id in range(num_users):

        user_profile_start = urm_test.indptr[user_id]
        user_profile_end = urm_test.indptr[user_id+1]

        relevant_items = urm_test.indices[user_profile_start:user_profile_end]

        if relevant_items.size == 0:
            num_users_skipped += 1
            continue

        recommendations = recommender.recommend(
            user_id=user_id, at=num_recommendations, urm_train=urm_train, remove_seen=True)

        cumulative_precision += precision(recommendations, relevant_items)
        cumulative_recall += recall(recommendations, relevant_items)
        cumulative_MAP += mean_average_precision(
            recommendations, relevant_items)

        num_users_evaluated += 1

        if len(relevant_items) > 0:

            recommended_items = recommender.recommend(
                user_id=user_id, urm_train=urm_train, at=num_recommendations)
            num_users_evaluated += 1

            cumulative_precision += precision(recommended_items,
                                              relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += mean_average_precision(
                recommended_items, relevant_items)

        cumulative_precision /= max(num_users_evaluated, 1)
        cumulative_recall /= max(num_users_evaluated, 1)
        cumulative_MAP /= max(num_users_evaluated, 1)

    return cumulative_precision, cumulative_recall, cumulative_MAP, num_users_evaluated, num_users_skipped


def hyperparameter_tuning():
    shrinks = [0, 1, 5, 10, 50]
    results = []
    for shrink in shrinks:
        print(f"Trying shrink {shrink}")

        itemknn_recommender = CFItemKNN(shrink=shrink)
        itemknn_recommender.fit(urm_train.tocsc(), similarity)

        ev_precision, ev_recall, ev_map, _, _ = evaluator(
            itemknn_recommender, urm_train, urm_validation)
        results.append((shrink, (ev_precision, ev_recall, ev_map)))
    return results


def prepare_submission(ratings: pd.DataFrame, users_to_recommend: np.array, urm_train: sp.csr_matrix, recommender: object):
    users_ids_and_mappings = ratings[ratings.user_id.isin(
        users_to_recommend)][["user_id", "mapped_user_id"]].drop_duplicates()
    items_ids_and_mappings = ratings[[
        "item_id", "mapped_item_id"]].drop_duplicates()

    mapping_to_item_id = dict(zip(ratings.mapped_item_id, ratings.item_id))

    recommendation_length = 10
    submission = []
    for idx, row in users_ids_and_mappings.iterrows():
        user_id = row.user_id
        mapped_user_id = row.mapped_user_id

        recommendations = recommender.recommend(user_id=mapped_user_id,
                                                urm_train=urm_train,
                                                at=recommendation_length,
                                                remove_seen=True)

        submission.append(
            (user_id, [mapping_to_item_id[item_id] for item_id in recommendations]))

    return submission


def write_submission(submissions):
    with open("./submission.csv", "w") as f:
        for user_id, items in submissions:
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")


ratings = load_data("data_train")
ratings, num_users, num_items = preprocess_data(ratings)
urm_train, urm_validation, urm_test = dataset_splits(
    ratings, num_users, num_items, validation_percentage=0.10, testing_percentage=0.20)
weights = similarity(urm_train.tocsc(), shrink=5)

#itemknn_recommender = CFItemKNN(shrink=50)
#itemknn_recommender.fit(urm_train.tocsc(), similarity)

"""
for user_id in range(10):
    print(itemknn_recommender.recommend(user_id=user_id,
                                        urm_train=urm_train, at=10, remove_seen=True))
"""

# cumulative_precision, cumulative_recall, cumulative_MAP, num_users_evaluated, num_users_skipped = evaluator(
# itemknn_recommender, urm_train, urm_test)

# print('cumulative_precision', cumulative_precision)
# print('cumulative_recall', cumulative_recall)
# print('cumulative_MAP', cumulative_MAP)
# print('num_users_evaluated', num_users_evaluated)
# print('num_users_skipped', num_users_skipped)


# print(hyperparameter_tuning())

best_shrink = 50
urm_train_validation = urm_train + urm_validation
best_recommender = CFItemKNN(shrink=best_shrink)
best_recommender.fit(urm_train_validation.tocsc(), similarity)
users_to_recommend = np.random.choice(
    ratings.user_id.unique(), size=urm_train_validation.shape[0], replace=False)

submission = prepare_submission(
    ratings, users_to_recommend, urm_train_validation, best_recommender)

write_submission(submission)
