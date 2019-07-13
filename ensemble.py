class Ensemble:

    def __init__(self, label='item_recommendations', strategy='voting'):
        self.label = label
        self.strategy = strategy

    def fit(self, data):
        self.submissions = data
        self.no_submissions = len(self.submissions)
        self.n = self.submissions[0].shape[0]
        self.prediction = self.submissions[0].copy()
        self.prediction.drop([self.label], inplace=True, axis=1)

        # create new dataframe
        self.labels = pd.DataFrame()

        # copy all label columns to the new dataframe
        for idx, submission in enumerate(self.submissions):
            self.labels.insert(idx, idx, submission[self.label])

    def predict(self):
        ranked_impressions = list()

        # rows
        for row in self.labels.itertuples():

            if self.strategy == 'voting':
                ranked_impressions.append(self.voting(row))

        # set resulting column
        self.prediction[self.label] = ranked_impressions

        return self.prediction

    def voting(self, row):
        # create empty dict for storying the merged item impressions
        items_dict = {}

        # cols
        for col_index in range(self.no_submissions):

            # + 1 because the inded is the first
            items = row[col_index + 1].split(" ")

            for idx, item in enumerate(items):
                # add or update value by adding the inverse rank as a score
                items_dict[item] = items_dict.get(item, 0) + 1 / (idx + 1)

        # sort final dict and return row
        sorted_impression_list = sorted(items_dict, key=items_dict.get, reverse=True)
        # convert impression list to one string
        impressions_as_string = " ".join(str(x) for x in sorted_impression_list)

        return impressions_as_string