import pandas as pd
import numpy as np
from scipy import sparse as sp 
import sklearn
import sklearn.preprocessing as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:

    def build_item_contents(self):

        vectorizer = CountVectorizer()
        self.tfidf = vectorizer.fit_transform(self.metadata['properties'])

    def get_item_vectors(self, item_ids):
        item_idx = [self.item_ids.index(item_id) for item_id in item_ids]
        item_vector = self.tfidf[item_idx]
        return item_vector


    def get_user_profile(self, user_id, ratings):
        user_session_ids = np.array( ratings.loc[ ratings['user_id'] == user_id ]['session_id'])
        user_ratings = np.array( ratings.loc[ ratings['user_id'] == user_id ]['weight'] )

        td_idf_vectors = self.get_item_vectors(user_session_ids)
        user_profile = sp.csr_matrix(td_idf_vectors).multiply(sp.csr_matrix(user_ratings).T).mean(axis=0)

        user_profile = pp.normalize(user_profile)

        return user_profile



    def build_user_profiles(self):
        positive_ratings = self.ratings[self.ratings['weight']>2]
        self.user_profiles = {}
        for user_id in positive_ratings['user_id'].unique():
            self.user_profiles[user_id] = self.get_user_profile(user_id, positive_ratings)


    def recommend(self, user_id, choice_ids=None, topN=20):
        user_session_ids = self.ratings.loc[ self.ratings['user_id'] == user_id ]['session_id'].tolist()
        num_rated = len(user_session_ids)

        if choice_ids is None:
            choice_ids = self.itemsIds

        user_profile = self.get_user_profile(user_id, self.ratings)
        sims = linear_kernel(user_profile,self.tfidf, dense_output = True)
        sims_sorted = np.argsort(sims)[0][::-1][0:topN+num_rated]
        recommendations = np.array(self.references)[sims_sorted]  
        return recommendations




    def build_model(self, ratings, metadata):
        self.ratings = ratings
        self.metadata = metadata
        self.itemsIds = self.ratings.session_id.unique()
        self.itemsIds.sort()
        self.userIds = self.ratings.user_id.unique()
        self.userIds.sort()
        self.item_ids = self.ratings['session_id'].tolist()
        self.references = self.metadata['item_id']

        self.build_item_contents()
        self.build_user_profiles()
