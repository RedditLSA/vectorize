from __future__ import division

import csv
import os
from itertools import islice

import numpy as np
import sys
from scipy.sparse import csc_matrix, csr_matrix, save_npz
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


class PpmiTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        N = np.sum(X)
        P_ab = X / float(N)
        P_a = np.sum(P_ab, axis=1)
        P_b = np.sum(P_ab, axis=0)
        X = np.log2(P_ab / P_a[:, None] / P_b)
        X[X <= 0] = 0
        return X


def main():
    n_points = 16478239

    sub_to_index = {}
    index_to_sub = {}
    ref_to_index = {}
    data = []
    row_ind = []
    col_ind = []

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    print('reading rows')

    with open(os.path.join(in_dir, 'sub_ref_overlap_all_starting_2015_01.csv')) as f:
        for point_index, (sub, ref, n_shared_authors) in enumerate(islice(csv.reader(f), 1, None)):
            n_shared_authors = int(n_shared_authors)
            try:
                sub_index = sub_to_index[sub]
            except KeyError:
                sub_index = len(sub_to_index)
                sub_to_index[sub] = sub_index
                index_to_sub[sub_index] = sub
            try:
                ref_index = ref_to_index[ref]
            except KeyError:
                ref_index = len(ref_to_index)
                ref_to_index[ref] = ref_index
            data.append(n_shared_authors)
            row_ind.append(sub_index)
            col_ind.append(ref_index)

            if (point_index + 1) % (n_points // 100) == 0:
                print('{:.1%}'.format((point_index + 1) / n_points))

    print('Transforming data')
    ppmi = PpmiTransformer()
    normalizer = Normalizer(norm='l2', copy=False)
    pipeline = make_pipeline(
        ppmi,
        normalizer,
    )
    overlap = csc_matrix((data, (row_ind, col_ind)), shape=(len(sub_to_index), len(ref_to_index))).toarray()
    X = pipeline.fit_transform(overlap)

    print('Fetching most popular subs')
    subs_by_popularity = []

    with open(os.path.join(in_dir, 'sub_pop_all_starting_2015_01.csv')) as f:
        for sub, n_authors in islice(csv.reader(f), 1, None):
            subs_by_popularity.append((sub, int(n_authors)))

    print('Saving subs_by_popularity')
    joblib.dump(subs_by_popularity, os.path.join(out_dir, 'subs_by_popularity.pkl'), protocol=2)
    print('Saving sub_to_index')
    joblib.dump(sub_to_index, os.path.join(out_dir, 'sub_to_index.pkl'), protocol=2)
    print('Saving index_to_sub')
    joblib.dump(index_to_sub, os.path.join(out_dir, 'index_to_sub.pkl'), protocol=2)
    print('Saving X')
    X = csc_matrix(X)
    save_npz(os.path.join(out_dir, 'X.npz'), X)


if __name__ == '__main__':
    main()
