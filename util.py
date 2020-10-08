import pandas
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

def pre_process(train_raw, test_raw):
    """ 
    @brief:= Pre process the data to normalize
    @in:= train_raw: raw train data
          test_raw: raw test data
    @out:= processed data with continuous fields 
           normalized btw 0 and 1, and binary encoded
           for the features "US" and "Urban" and
           one-hot encoding for "ShelveLoc"
    """

    features = train_raw.columns[1:] #ignoring sales

    for feature in features:
        if feature in ["Urban", "US"]:
            train_raw[feature] = LabelBinarizer().fit_transform(train_raw[feature])
            test_raw[feature] = LabelBinarizer().fit_transform(test_raw[feature])
            continue

        if feature == "ShelveLoc":
            train_raw = pandas.concat([train_raw,pandas.get_dummies(train_raw['ShelveLoc'], 
                                 prefix='ShelveLoc',dummy_na=False)],axis=1).drop(['ShelveLoc'],axis=1)
            test_raw = pandas.concat([test_raw,pandas.get_dummies(test_raw['ShelveLoc'], 
                                 prefix='ShelveLoc',dummy_na=False)],axis=1).drop(['ShelveLoc'],axis=1)
            continue

        mu = train_raw[feature].mean()
        sig = train_raw[feature].std(ddof=0)
        train_raw[feature] = (train_raw[feature] - mu) / sig
        test_raw[feature] = (test_raw[feature] - mu) / sig

    return train_raw, test_raw



