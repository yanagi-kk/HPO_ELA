import pandas as pd
from sklearn.preprocessing import LabelEncoder

class preprocess_data:
    def __init__(self, data_name):
        self.data_name = data_name

    def get_target(self):
        target = {
            "wilt": "Class",
            "ada": "'class'",
            "analcatdata_dmft": "Prevention",
            "bank-marketing": "Class",
            "cylinder-bands": "'band_type'"
        }
        return target[self.data_name]
    
    def LabelEncoding(self, df):
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])
        return df
    
    def get_X_y(self):
        df = pd.read_csv("../data/" + self.data_name + ".csv")
        df = self.LabelEncoding(df)

        target = self.get_target()
        X = df.drop(["id", target], axis=1)
        y = df[target]
        if self.data_name in ["wilt", "bank-marketing"]:
            y = y.apply(lambda x: x-1)

        return X, y
    

