import datetime
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set(font_scale=1.2)


def order_features_by_gains(model, importance_type='weight'):
    sorted_feats = sorted(model.get_score(importance_type=importance_type).items(), key=lambda k: -k[1])
    return sorted_feats


def get_feature_importance(fnames, model, importance_type='weight'):
    # Feature importance values
    model_fi = pd.DataFrame(fnames)
    model_fi.columns = ['feature']
    ff_gain = order_features_by_gains(model, importance_type=importance_type)
    ff = np.zeros(len(fnames))
    for k,v in ff_gain:
        ff[fnames.index(k)] = v
    model_fi['importance'] = 100.0 * (ff / ff.max())
    return model_fi.sort_values('importance', ascending=0)


def draw_feature_importance(model_fi, topn):
    pos = np.arange(topn)[::-1] + 1
    topn_features = list(model_fi.sort_values('importance', ascending=0)['feature'].head(topn))

    plt.figure(figsize=(6, 6))
    plt.axis([0, 100, 0, topn+1])
    plt.barh(pos, model_fi.sort_values('importance', ascending=0)['importance'].head(topn), align='center')
    plt.yticks(pos, topn_features)

    plt.xlabel('Relative Importance')
    #plt.grid()
    plt.title('Model Feature Importance Plot', fontsize=20)
    plt.show()
