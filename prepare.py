import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from acquire import get_titanic_data, get_iris_data
import pymysql

def prep_iris():
    df = get_iris_data()
    df = df.drop(columns='species_id')
    dummy = pd.get_dummies(df.species, dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy], axis=1)
    return df