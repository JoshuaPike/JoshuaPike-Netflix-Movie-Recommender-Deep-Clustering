import pandas as pd
import numpy as np
# train_df = pd.read_csv('/content/drive/My Drive/Colab_Notebooks/Assignments/Deep_Clustering/dataset/train.csv') # Change this to where our training set is
train_df = pd.read_csv('train.csv')
# print(train_df)
df_p = pd.pivot_table(train_df,values='rating',index='customer-id',columns='movie-id')
# print(df_p)

# print(train_df)
# print(train_df[['customer-id', 'movie-id']])
# if (train_df[['customer-id', 'movie-id']].values == [24294, 1262]).all(axis=1).any():
#     print('fuck yeah')



# Fill in missing values in a meaningful way
for movie_id in df_p.columns:
    mode = df_p[movie_id].mode()[0]
    df_p[movie_id].fillna(mode, inplace=True)


N, K = df_p.shape
data = {
    "rating": df_p.to_numpy().ravel("F"),
    "movie-id": np.asarray(df_p.columns).repeat(N),
    "customer-id": np.tile(np.asarray(df_p.index), K),
}
filled_df = pd.DataFrame(data, columns=["customer-id", "movie-id", "rating"])

# print(filled_df)

# print(fuck.loc[fuck['customer-id'] == 1262 and fuck['movie-id'] == 24294])
# print(fuck[(fuck[['movie-id','customer-id']].values == [1262, 24294]).all(axis=1)])
# print(fuck.query(1262 == 'customer-id' and 24294 == 'movie-id'))

# movie_col = []
# user_col = []
# rating_col = []

# print(df_p.shape)

# for user_id in df_p.index:
#     for movie_id in df_p.columns:
#         user_col.append(user_id)
#         movie_col.append(movie_id)
#         rating_col.append(df_p[movie_id][user_id])

# d = {'customer-id': user_col, 'movie-id': movie_col, 'rating': rating_col}
# df = pd.DataFrame(data=d)


# df_p = df_p.to_numpy()
# Rows (users/user_ids) and cols (movies/movie_ids) are in numerical order.

# !pip install scikit-surprise #  colaborative filtering packages (svd, mf, co-clustering, etc) 
# !pip install scikit-learn==0.22.2 --upgrade

import math
import re
from tqdm import tqdm
from scipy.sparse import csr_matrix
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# matrix_factorization.SVD: https://surprise.readthedocs.io/en/stable/matrix_factorization.html
reader = Reader()
svd = SVD(n_factors=5)
# data = Dataset.load_from_df(train_df[['customer-id', 'movie-id', 'rating']][:], reader)
data = Dataset.load_from_df(filled_df[['customer-id', 'movie-id', 'rating']][:], reader)
full_train = data.build_full_trainset()
svd.fit(full_train) 

np.save('svd_pu.npy', svd.pu)

# for user in range(full_train.n_users):
#     for item in range(full_train.n_items):
#         # .to_raw_uid/iid gets back original id
#         # for each (u, i) in df_p (sparse matrix) add bu + bi + qi.transpose * pu
#         # print(str(user) + '   ' + str(item))
#         pred = bu[user] + bi[item] + (np.matmul(np.array(qi[item]).transpose(), np.array(pu[user]))) # Scalar

#         df_p[user, item] += pred
#         # print(df_p[user, item])

# print(df_p)

# np.save('svd_mat.npy', df_p)

# scikit-learn k means clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
