import os
from typing import Tuple, Callable, Dict, Optional, List
import numpy as np
import pandas as pd
import scipy.sparse as sp
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

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

    # print('num users {}. From: {} to {}'.format(num_users, min_user_id, max_user_id))
    # print('num items {}. From: {} to {}'.format(num_items, min_item_id, max_item_id))

    ratings = pd.merge(left=ratings, right=mapping_user_id, how="inner", on="user_id")
    ratings = pd.merge(left=ratings, right=mapping_item_id, how="inner", on="item_id")

    return ratings, num_users, num_items


def preprocess_icm(importances: pd.DataFrame):
    unique_items = importances.item_id.unique()
    unique_features = importances.feature_id.unique()

    num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()
    num_features, min_feature_id, max_feature_id = unique_features.size, unique_features.min(
    ), unique_features.max()

    mapping_item_id = pd.DataFrame({
        "mapped_item_id": np.arange(num_items),
        "item_id": unique_items
    })

    mapping_feature_id = pd.DataFrame({
        "mapped_feature_id": np.arange(num_features),
        "feature_id": unique_features
    })

    importances = pd.merge(
        left=importances, right=mapping_item_id, how="inner", on="item_id")
    importances = pd.merge(
        left=importances, right=mapping_feature_id, how="inner", on="feature_id")

    return importances, num_items, num_features


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
        test_size=validation_percentage)

    urm_train = sp.csr_matrix((ratings_training, (user_ids_training, item_ids_training)), shape=(num_users, num_items))

    urm_validation = sp.csr_matrix((ratings_validation, (user_ids_validation, item_ids_validation)), shape=(num_users, num_items))

    urm_test = sp.csr_matrix((ratings_test, (user_ids_test, item_ids_test)), shape=(num_users, num_items))

    return urm_train, urm_validation, urm_test


def matrix_similarity(shrink: int):
    icm_csv = pd.read_csv("./data/data_ICM_title_abstract.csv",
                          names=["item_id", "feature_id", "importance"],
                          header=0,
                          dtype={
                              "item_id": np.int32,
                              "feature_id": np.int32,
                              "importance": np.longdouble
                          })
    icm, num_items, num_features = preprocess_icm(icm_csv)
    icm = icm[['importance', 'mapped_item_id', 'mapped_feature_id']]
    matrix = pd.pivot_table(icm, values="importance",index="mapped_item_id", columns="mapped_feature_id")
    matrix = matrix.fillna(0)
    item_similarity = 1-pairwise_distances(matrix, metric='cosine')
    return item_similarity


class CFItemKNN(object):
    def __init__(self, shrink: int):
        self.shrink = shrink
        self.weights = None

    def fit(self, urm_train: sp.csc_matrix, similarity_function):
        if not sp.isspmatrix_csc(urm_train):
            raise TypeError(
                f"We expected a CSC matrix, we got {type(urm_train)}")

        self.weights = similarity_function(self.shrink)

    def recommend(self, user_id: int, urm_train: sp.csr_matrix, at: Optional[int] = None, remove_seen: bool = True):
        user_profile = urm_train[user_id]

        items = np.zeros(self.weights.shape[0])
        for item_id in urm_train[user_id].indices:
            items[item_id] = 1

        print('items array dimension', items.shape)
        print('weights matrix dimension', self.weights.shape)

        ranking = items.dot(self.weights)

        if remove_seen:
            user_profile_start = urm_train.indptr[user_id]
            user_profile_end = urm_train.indptr[user_id+1]
            seen_items = urm_train.indices[user_profile_start:user_profile_end]
            ranking[seen_items] = -np.inf

        ranking = np.flip(np.argsort(ranking))
        return ranking[:at]


def recall(recommendations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant) / relevant_items.shape[0]

    return recall_score


def precision(recommendations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant) / recommendations.shape[0]

    return precision_score


def mean_average_precision(recommendations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

    precision_at_k = is_relevant * \
        np.cumsum(is_relevant, dtype=np.float32) / \
        (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(precision_at_k) / \
        np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluator(recommender: object, urm_train: sp.csr_matrix, urm_test: sp.csr_matrix):
    recommendation_length = 10
    accum_precision = 0
    accum_recall = 0
    accum_map = 0

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

        recommendations = recommender.recommend(user_id=user_id,
                                                at=recommendation_length,
                                                urm_train=urm_train,
                                                remove_seen=True)

        accum_precision += precision(recommendations, relevant_items)
        accum_recall += recall(recommendations, relevant_items)
        accum_map += mean_average_precision(recommendations, relevant_items)

        num_users_evaluated += 1

    accum_precision /= max(num_users_evaluated, 1)
    accum_recall /= max(num_users_evaluated, 1)
    accum_map /= max(num_users_evaluated, 1)

    return accum_precision, accum_recall, accum_map, num_users_evaluated, num_users_skipped


def hyperparameter_tuning():
    shrinks = [0, 1, 5, 10, 50]
    results = []
    for shrink in shrinks:
        print(f"Currently trying shrink {shrink}")

        itemknn_recommender = CFItemKNN(shrink=shrink)
        itemknn_recommender.fit(urm_train.tocsc(), matrix_similarity)

        ev_precision, ev_recall, ev_map, _, _ = evaluator(
            itemknn_recommender, urm_train, urm_validation)

        results.append((shrink, (ev_precision, ev_recall, ev_map)))

    return results


def prepare_submission(ratings: pd.DataFrame, users_to_recommend: np.array, urm_train: sp.csr_matrix, recommender: object):
    users_ids_and_mappings = ratings[ratings.user_id.isin(users_to_recommend)][["user_id", "mapped_user_id"]].drop_duplicates()
    items_ids_and_mappings = ratings[["item_id", "mapped_item_id"]].drop_duplicates()

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
    now = datetime.datetime.now()
    with open("./submission-"+(now.strftime("%Y%m%d%H%M%S"))+".csv", "w") as f:
        f.write(f"user_id,item_list\n")
        for user_id, items in submissions:
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")

ratings = load_data("data_train")
ratings, num_users, num_items = preprocess_data(ratings)

urm_train, urm_validation, urm_test = dataset_splits(ratings, num_users, num_items, validation_percentage=0.10, testing_percentage=0.20)
urm_csc = urm_train.tocsc()

# itemknn_recommender = CFItemKNN(shrink=50)
# itemknn_recommender.fit(urm_train.tocsc(), matrix_similarity)
# accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped = evaluator(itemknn_recommender, urm_train, urm_test)

"""
hyperparameter_results = hyperparameter_tuning()

print(hyperparameter_results)
[
    (0, (0.009005002779321885, 0.046096116701285765, 0.019075018283722222)),
    (1, (0.011367426347971166, 0.057408502450324514, 0.02265875431761243)),
    (5, (0.01489716509171774, 0.07435567616325088, 0.031977590080769065)),
    (10, (0.016008893829905637, 0.07969989824477562, 0.033744143571825605)),
    (50, (0.017565314063368687, 0.09009086005771763, 0.036339536406455215)) <- this is the best
]
"""

best_shrink = 50
urm_train_validation = urm_train + urm_validation

best_recommender = CFItemKNN(shrink=best_shrink)
best_recommender.fit(urm_train_validation.tocsc(), matrix_similarity)

users_to_recommend = pd.read_csv("./data/data_target_users_test.csv")['user_id'].values

submission = prepare_submission(ratings, users_to_recommend, urm_train_validation, best_recommender)

write_submission(submission)
