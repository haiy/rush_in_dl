import requests
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split

'''

'''

iris_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
result = requests.get(iris_data_url).text

def get_raw_data():
    return pd.read_csv(StringIO(result))


def get_train_test_data():
    raw_df = get_raw_data()
    train, test = train_test_split(raw_df, 0.3)
    return train, test





