import pandas as pd
import numpy as np
from scipy import sparse as sp


class UUCFRecommender:
    UU = {}  # session-session similarities dict

    def __init__(self, with_abs_sim=True, with_deviations=True, k=5, ratings=None, R=None, session_id_labels=None, session_id_levels=None, reference_labels=None,
                              reference_levels=None):
        self.with_abs_sim = with_abs_sim
        self.with_deviations = with_deviations
        self.k = 5
        self.ratings = ratings
        self.R = R
        self.session_id_labels = session_id_labels
        self.session_id_levels = session_id_levels
        self.reference_labels = reference_labels
        self.reference_levels = reference_levels


    def create_Ratings_Matrix(self):
        """
            Create the rating matrix and assign a reference id for each session_id and item id (reference)
        """

        self.items = self.ratings['reference'].unique()
        self.items.sort()

        self.session_ids = self.ratings['session_id'].unique()
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

    def compute_session_avgs(self):
        """
            Computes the average rating of each session
        """

        session_sums = self.R.sum(axis=1).A1  ## matrix converted to 1-D array via .A1
        self.session_cnts = (self.R != 0).sum(axis=1).A1
        self.session_avgs = session_sums / self.session_cnts

    def compute_pairwise_session_similarity(self, s1_id, s2_id):

        """
           Computes the mean-centered cosine similarity between two sessions

           Parameters
           ----------
            s1_id: str
                ID corresponding to session 1
            s2_id: str
                ID corresponding to session 2

           Returns
           -------
           Similarity: Float
                Similarity value between session 1 and session 2

        """

        # Copy the vector corresponding to s1 and s2
        s1 = self.R[s1_id, :].copy()
        s2 = self.R[s2_id, :].copy()

        # normalize values with session weight mean
        s1.data = s1.data - self.session_avgs[s1_id]
        s2.data = s2.data - self.session_avgs[s2_id]
        numerator = s1.dot(s2.T)

        # calculate sum of squares for s1 and s2
        ssq_s1 = np.sum(s1.data ** 2)
        ssq_s2 = np.sum(s2.data ** 2)

        denominator = np.sqrt(ssq_s1 * ssq_s2)

        if denominator == 0:
            similarity = 0.;
        else:
            similarity = numerator / denominator

        return similarity

    def compute_session_similarities(self, s_id):

        """
           Computes the mean-centered cosine similarity between two sessions

           Parameters
           ----------
            s1_id: str
                ID corresponding to the session
        """

        if s_id in self.UU.keys():
            return

        sS = np.empty((self.m,))

        nnz_per_row = np.diff(self.R.indptr)
        avg_array = np.repeat(self.session_avgs, nnz_per_row)
        R_copy = self.R.copy()
        R_copy.data -= avg_array

        s1 = R_copy[s_id, :].copy()

        numerator = R_copy.dot(s1.T)

        ssq_s1 = np.sum(s1.data ** 2)

        R_copy.data = R_copy.data ** 2
        ssq_s2 = R_copy.sum(axis=1)

        denominator = np.sqrt(ssq_s2.dot(ssq_s1))

        sS = numerator / denominator;
        sS = np.nan_to_num(sS)

        self.UU[s_id] = sS

    def create_session_neighborhood(self, s_id, i_id):

        """
           Creates the neighborhood of a given session

           Parameters
           ----------
            s_id: str
                ID corresponding to the session
            i_id: str
                ID corresponding to the item

            Returns
            -------
            nh: dict
                Neighborhood corresponding to the session s_id
        """

        nh = {}  # Initialize the beighborhood dict

        self.compute_session_similarities(s_id)
        uU = self.UU[s_id].copy()

        uU_copy = uU.copy()

        i = 0
        SessionArray = np.squeeze(np.asarray(uU))
        argArray = np.argsort(SessionArray)[::-1]

        # remove the session from the sessions list
        index = np.argwhere(argArray == s_id)
        argArray = np.delete(argArray, index)

        if self.with_abs_sim == False:
            for x in argArray:
                if (x, i_id) in self.R_dok:
                    nh[x] = SessionArray[x]
                    i += 1
                if i == self.k:
                    break
        else:
            tempArray = np.absolute(SessionArray)
            absArray = np.argsort(tempArray)[::-1]
            for x in absArray:
                if (x, i_id) in self.R_dok:
                    nh[x] = SessionArray[x]
                    i += 1
                if i == self.k:
                    break

        return nh

    def predict_rating(self, s_id, i_id):

        """
        Predicts the rating session s_id would give to item i_id

        Parameters
        ----------
        s_id: str
           ID corresponding to the session
        i_id: str
           ID corresponding to the item

        Returns
        -------
        prediction: float
           Predicted rating
        """


        nh = self.create_session_neighborhood(s_id, i_id)

        neighborhood_weighted_avg = 0.

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

        if self.with_deviations:
            prediction = self.session_avgs[s_id] + neighborhood_weighted_avg
        else:
            prediction = neighborhood_weighted_avg
        return prediction


    def recommend(self, session_id, item_ids=None, topN=20):
        """
        Recomment topN items to a session

        Parameters
        ----------
        session_id: str
           ID corresponding to the session
        topN: int
           Maximum number of items to recommend
        item_ids: list
            list of items to consider when recommending

        Returns
        -------
        recommendations: list
           List of item recommendations
        """
        items_viewed_in_session = self.ratings[self.ratings['session_id'] == session_id]['reference'].tolist()

        s_id = self.sessionId_to_sessionIDX[session_id]

        recommendations = []

        if item_ids is None:
            item_ids = self.items

        for item_id in item_ids:
            if item_id in items_viewed_in_session:
                continue
            i_id = self.itemId_to_itemIDX[item_id]

            rating = self.predict_rating(s_id, i_id)
            recommendations.append((item_id, rating))

        recommendations = sorted(recommendations, key=lambda x: -x[1])[:topN]

        return [item_id for item_id, rating in recommendations]
