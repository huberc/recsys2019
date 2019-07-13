import pandas as pd

class Ensemble:

    def __init__(self, submission_files):
        self.submission_files = submission_files
        self.nb_sub = len(self.submission_files)
        self.n = self.submission_files[0].shape[0]
        self.prediction = self.submission_files[0].copy()
        self.prediction.drop(["item_recommendations"], inplace=True, axis=1)

    def predict(self):

        self.labels = pd.DataFrame()

        for idx, submission in enumerate(self.submission_files):
            self.labels.insert(idx, idx, self.submission_files["item_recommendations"])

        ranked_impressions = list()

        for row in self.labels.itertuples():

            if self.strategy == 'voting':
                ranked_impressions.append(self.voting(row))

        self.prediction["item_recommendations"] = ranked_impressions

        return self.prediction

    def voting(self, row):

        items_dict = {}

        for col_index in range(self.nb_sub):

            items = row[col_index + 1].split(" ")

            for idx, item in enumerate(items):

                items_dict[item] = items_dict.get(item, 0) + 1 / (idx + 1)


        sorted_impression_list = sorted(items_dict, key=items_dict.get, reverse=True)

        impressions_as_string = " ".join(str(x) for x in sorted_impression_list)

        return impressions_as_string