import pandas as pd
import numpy as np
from scipy import sparse as sp 
import sklearn
import sklearn.preprocessing as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:

    def __init__(self, profile_type='plot'):
        self.profile_type = profile_type

    def get_movie_titles(self, ids):
        return [ self.items[self.items['movieId'] == id]['title'].item() for id in ids]


    def build_item_contents(self):

        vectorizer = TfidfVectorizer(stop_words='english') # Define a TF-IDF Vectorizer that removes all english stop words (e.g., 'the', 'a')

        ########## BEGIN HERE ##########
        self.plot_tfidf = vectorizer.fit_transform(self.items['plot'])
        self.plot_tfidf_tokens = vectorizer.get_feature_names()
    
        self.meta_tfidf = vectorizer.fit_transform(self.items['metadata'])
        self.meta_tfidf_tokens =  vectorizer.get_feature_names()
        ##########  END HERE  ##########

        self.set_content_type()


    def set_content_type(self):
        if self.profile_type == 'plot':
            self.tfidf = self.plot_tfidf
            self.tfidf_tokens = self.plot_tfidf_tokens
        else:
            self.tfidf = self.meta_tfidf
            self.tfidf_tokens = self.meta_tfidf_tokens


    def get_item_vectors(self, item_ids):
        item_idx = [self.item_ids.index(item_id) for item_id in item_ids]
        item_vector = self.tfidf[item_idx]
        return item_vector


    def get_user_profile(self, user_id, ratings):
        user_rated_item_ids = np.array( ratings.loc[ ratings['user'] == user_id ]['item'] )
        user_ratings = np.array( ratings.loc[ ratings['user'] == user_id ]['rating'] )

        ########## BEGIN HERE ##########
        td_idf_vectors = self.get_item_vectors(user_rated_item_ids)
        user_profile = sp.csr_matrix(td_idf_vectors).multiply(sp.csr_matrix(user_ratings).T).mean(axis=0)
        ##########  END HERE  ##########

        user_profile = pp.normalize(user_profile)

        return user_profile



    def build_user_profiles(self):
        positive_ratings = self.ratings[self.ratings['rating']>3]
        self.user_profiles = {}
        for user_id in positive_ratings['user'].unique():
            self.user_profiles[user_id] = self.get_user_profile(user_id, positive_ratings)


    ### **** NOTE: Slight change from Assignment 3. ****
    ### Recommends topN items to the user based on her/his profile.
    ### The recommendations should exclude items already rated by the user, and
    ### only items among choice_ids. If choice_ids is None, then all items are considered.
    ### Steps to implement:
    ### 1. Retrieve the user profile
    ### 2. Compute the cosine similarity between the user profile and each td-idf vector,
    ###    and store it into array `sims`. Tips: Use `linear_kernel` from scikit-learn to take the inner product,
    ###    since all vectors are normalized. Also, flatten the output at the end.
    ### 3. Identify the indices in `sims` that have the largest similarities.
    ###    Tips: `a[::-1]` returns the reverse of list `a`. You may want to use the `numpy.argsort` method.
    ### 4. Retrieve the item_ids from `self.item_ids` that correspond to the indices found.
    ### 5. Include in the recommendation list only items from choice_ids, and exclude those in user_rated_item_ids.
    ### 6. Return only the topN. Recommended items should be sorted from most to least similar to user profile.

    def recommend(self, user_id, choice_ids=None, topN=20):
        user_rated_item_ids = self.ratings.loc[ self.ratings['user'] == user_id ]['item'].tolist()
        num_rated = len(user_rated_item_ids)

        if choice_ids is None:
            choice_ids = self.itemsIds


        ########## BEGIN HERE ##########
        user_profile = self.get_user_profile(user_id, self.ratings)
        sims = linear_kernel(user_profile,self.tfidf, dense_output = True)
        #print("np.argsort(sims)[0]",np.argsort(sims)[0][::-1][0:50])
        #print(sims[0][32])
        #print(sims[0][10])
        #print(sims[0][16])
        #print(sims[0][37])
        #print(sims[0][0])
        #print(sims[0][89])
        #print(sims[0][70])
        sims_sorted = np.argsort(sims)[0][::-1]
        recommendations = np.array(self.item_ids)[sims_sorted]
        #print(recommendations[0:50])
        #print(user_rated_item_ids)
        #print(choice_ids)
        filter = np.isin(recommendations, user_rated_item_ids, invert=True) & np.isin(recommendations, choice_ids)
        recommendations = recommendations[filter][0:topN]
        ##########  END HERE  ##########

        return recommendations




    def build_model(self, ratings, items):
        self.ratings = ratings
        self.items = items
        self.itemsIds = self.ratings.item.unique()
        self.itemsIds.sort()
        self.userIds = self.ratings.user.unique()
        self.userIds.sort()
        self.item_ids = self.items['movieId'].tolist()

        self.build_item_contents()
        self.build_user_profiles()
