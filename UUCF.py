import pandas as pd
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import norm
import sklearn.preprocessing as pp


class UUCFRecommender:
    UU = {}  ## user-user similarities; constructed lazily

    def __init__(self, with_abs_sim=True, with_deviations=True, k=5, ratings=None, R=None, session_id_labels=None, session_id_levels=None, reference_labels=None,
                              reference_levels=None):
        self.with_abs_sim = with_abs_sim
        self.with_deviations = with_deviations
        self.k = 5
        self.ratings = ratings
        self.dataframe = ratings
        self.R = R
        self.session_id_labels = session_id_labels
        self.session_id_levels = session_id_levels
        self.reference_labels = reference_labels
        self.reference_levels = reference_levels


    def create_Ratings_Matrix(self):


        self.items = self.dataframe['reference'].unique()
        self.items.sort()

        self.session_ids = self.dataframe['session_id'].unique()
        self.session_ids.sort()

        self.m = self.session_ids.size

        ## Create dicts for sessions and items
        self.itemId_to_itemIDX = dict(zip(self.reference_levels, self.reference_labels))
        self.itemIDX_to_itemId = dict(zip(self.reference_labels, self.reference_levels))

        self.sessionId_to_sessionIDX = dict(zip(self.session_id_levels, self.session_id_labels))
        self.sessionIDX_to_sessionId = dict(zip(self.session_id_labels, self.session_id_levels))

        self.R = sp.csr_matrix((self.ratings['weight'], (self.ratings['session_id'].map(self.sessionId_to_sessionIDX),
                                                         self.ratings['reference'].map(self.itemId_to_itemIDX))))
        self.R_dok = self.R.todok()

    def compute_user_avgs(self):
        session_sums = self.R.sum(axis=1).A1  ## matrix converted to 1-D array via .A1
        self.session_cnts = (self.R != 0).sum(axis=1).A1
        self.session_avgs = session_sums / self.session_cnts

    def compute_pairwise_session_similarity(self, u_id, v_id):

        u = self.R[u_id, :].copy()
        v = self.R[v_id, :].copy()

        # normalize values with user rating mean
        u.data = u.data - self.session_avgs[u_id]
        v.data = v.data - self.session_avgs[v_id]
        numerator = u.dot(v.T)

        # calculate sum of squares
        ssq_u = np.sum(u.data ** 2)
        ssq_v = np.sum(v.data ** 2)

        denominator = np.sqrt(ssq_u * ssq_v)

        if denominator == 0:
            similarity = 0.;
        else:
            similarity = numerator / denominator

        return similarity

    def compute_session_similarities(self, u_id):
        if u_id in self.UU.keys():  ## persist
            return

        uU = np.empty((self.m,))

        ########## BEGIN HERE ##########
        nnz_per_row = np.diff(self.R.indptr)
        avg_array = np.repeat(self.session_avgs, nnz_per_row)
        R_copy = self.R.copy()
        R_copy.data -= avg_array

        u = R_copy[u_id, :].copy()

        numerator = R_copy.dot(u.T)

        # calculate sum of squares
        ssq_u = np.sum(u.data ** 2)

        R_copy.data = R_copy.data ** 2
        ssq_v = R_copy.sum(axis=1)

        denominator = np.sqrt(ssq_v.dot(ssq_u))

        uU = numerator / denominator;
        uU = np.nan_to_num(uU)
        ##########  END HERE  ##########

        """
        ########## BEGIN BONUS ##########

        ##########  END BONUS  ##########
        """

        self.UU[u_id] = uU

    def create_user_neighborhood(self, u_id, i_id):
        nh = {}  ## the neighborhood dict with (user id: similarity) entries
        ## nh should not contain u_id and only include users that have rated i_id; there should be at most k neighbors
        self.compute_session_similarities(u_id)
        uU = self.UU[u_id].copy()

        uU_copy = uU.copy()  ## so that we can modify it, but also keep the original

        ########## BEGIN HERE ##########
        i = 0
        userArray = np.squeeze(np.asarray(uU))
        argArray = np.argsort(userArray)[::-1]

        # remove user from userlist
        index = np.argwhere(argArray == u_id)
        argArray = np.delete(argArray, index)

        if self.with_abs_sim == False:
            for x in argArray:
                if (x, i_id) in self.R_dok:
                    nh[x] = userArray[x]
                    i += 1
                if i == self.k:
                    break
        else:
            tempArray = np.absolute(userArray)
            absArray = np.argsort(tempArray)[::-1]
            for x in absArray:
                if (x, i_id) in self.R_dok:
                    nh[x] = userArray[x]
                    i += 1
                if i == self.k:
                    break
        ##########  END HERE  ##########

        return nh

    def predict_rating(self, u_id, i_id):

        #         if (u_id, i_id) in self.R_dok:
        #             print("user", u_id, "has rated item", i_id, "with", self.R[u_id, i_id])
        #         else:
        #             print("user", u_id, "has not rated item", i_id)
        #         print("k:", self.k, "with_deviations:", self.with_deviations, "with_abs_sim:", self.with_abs_sim)

        nh = self.create_user_neighborhood(u_id, i_id)

        neighborhood_weighted_avg = 0.

        ########## BEGIN HERE ##########
        sumRatings = 0
        sumAbsWeights = 0
        for n_id, w in nh.items():
            if self.with_deviations:
                sumRatings += w * (self.R[n_id, i_id] - self.session_avgs[n_id])
                sumAbsWeights += abs(w)
            else:
                sumRatings += w * self.R[n_id, i_id]
                sumAbsWeights += abs(w)

        if np.isnan(sumAbsWeights) or np.isnan(sumRatings):
            neighborhood_weighted_avg = 0;
        else:
            neighborhood_weighted_avg = sumRatings / sumAbsWeights
        ##########  END HERE  ##########

        # if sum_weights != 0: ## avoid division by zero
        #    neighborhood_weighted_avg = sum_scores/sum_weights

        if self.with_deviations:
            prediction = self.session_avgs[u_id] + neighborhood_weighted_avg
        #             print("prediction ", prediction, " (user_avg ", self.user_avgs[u_id], " offset ", neighborhood_weighted_avg, ")", sep="")
        else:
            prediction = neighborhood_weighted_avg
        #             print("prediction ", prediction, " (user_avg ", self.user_avgs[u_id], ")", sep="")

        return prediction

    def recommend(self, session_id, item_ids=None, topN=20):

        movies_rated_by_user = self.ratings[self.ratings['session_id'] == session_id]['reference'].tolist()

        u_id = self.sessionId_to_sessionIDX[session_id]

        recommendations = []

        if item_ids is None:
            item_ids = self.items

        for item_id in item_ids:
            if item_id in movies_rated_by_user:
                continue
            i_id = self.itemId_to_itemIDX[item_id]

            rating = self.predict_rating(u_id, i_id)
            recommendations.append((item_id, rating))

        recommendations = sorted(recommendations, key=lambda x: -x[1])[:topN]
        #         print(recommendations)
        return [item_id for item_id, rating in recommendations]