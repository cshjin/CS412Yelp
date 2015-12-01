# -*- coding: utf-8 -*-
import json
import pandas as pd
from sklearn import decomposition, cross_validation
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder, Imputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import GaussianNB

import scipy.sparse
import cPickle as pickle

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from collections import OrderedDict

def clean_user_data():
    with open('../data/yelp_academic_dataset_user.json') as infile:
        dataset = {}
        for line in infile:
            data = json.loads(line)
            t_data = {}
            # t_data['user_id'] = data['user_id']
            t_data['yelping_since'] = 2015 - int(data['yelping_since'][0:4])
            t_data['votes_funny'] = data['votes']['funny']
            t_data['votes_useful'] = data['votes']['useful']
            t_data['votes_cool'] = data['votes']['cool']
            t_data['review_count'] = data['review_count']
            t_data['fans'] = data['fans']
            t_data['friends'] = len(data['friends'])
            t_data['average_stars'] = data['average_stars']
            t_data['elite'] = len(data['elite'])
            cp = data.get('compliments')
            if cp is not None:
                t_data['comp_profile'] = cp.get('profile')
                t_data['comp_cute'] = cp.get('cute')
                t_data['comp_funny'] = cp.get('funny')
                t_data['comp_plain'] = cp.get('plain')
                t_data['comp_writer'] = cp.get('writer')
                t_data['comp_note'] = cp.get('note')
                t_data['comp_photos'] = cp.get('photos')
                t_data['comp_hot'] = cp.get('hot')
                t_data['comp_more'] = cp.get('more')
                t_data['comp_cool'] = cp.get('cool')
                t_data['comp_list'] = cp.get('list')
            # dataset.append(t_data)
            dataset[data.get('user_id')] = t_data
        with open('../data/dataset_user_v2.json', 'w') as outfile:
            json.dump(dataset, outfile, indent=2)


def clean_busi_data():
    with open('../data/yelp_academic_dataset_business.json') as infile:
        dataset = {}
        for line in infile:
            data = json.loads(line)
            if 'Restaurants' in data['categories']:
                t_data = {}
                # t_data['business_id'] = data['business_id']
                t_data['longitude'] = data['longitude']
                t_data['latitude'] = data['latitude']
                t_data['review_count'] = data['review_count']
                t_data['stars'] = data['stars']
                t_data['open'] = data['open']
                t_data['city'] = data['city']
                t_data['state'] = data['state']

                # attributes
                # t_data['Accepts Credit Cards'] = data.get('attributes', None).get('Accepts Credit Cards', None)
                t_data['Accepts Insurance'] = data.get('attributes', None).get('Accepts Insurance', None)
                t_data['Ages Allowed'] = data.get('attributes', None).get('Ages Allowed', None)
                t_data['Alcohol'] = data.get('attributes', None).get('Alcohol', None)
                am = data.get('attributes', None).get('Ambience', None)
                if am is not None:
                    t_data['attr_romantic'] = data.get('attributes', None).get('Ambience').get('romantic', None)
                    t_data['attr_intimate'] = data.get('attributes', None).get('Ambience').get('intimate', None)
                    t_data['attr_classy'] = data.get('attributes', None).get('Ambience').get('classy', None)
                    t_data['attr_hipster'] = data.get('attributes', None).get('Ambience').get('hipster', None)
                    t_data['attr_divey'] = data.get('attributes', None).get('Ambience').get('divey', None)
                    t_data['attr_touristy'] = data.get('attributes', None).get('Ambience').get('touristy', None)
                    t_data['attr_trendy'] = data.get('attributes', None).get('Ambience').get('trendy', None)
                    t_data['attr_upscale'] = data.get('attributes', None).get('Ambience').get('upscale', None)
                    t_data['attr_casual'] = data.get('attributes', None).get('Ambience').get('casual', None)
                t_data['Attire'] = data.get('attributes', None).get('Attire', None)
                t_data['By Appointment Only'] = data.get('attributes', None).get('By Appointment Only', None)
                t_data['BYOB'] = data.get('attributes', None).get('BYOB', None)
                t_data['BYOB/Corkage'] = data.get('attributes', None).get('BYOB/Corkage', None)
                t_data['Caters'] = data.get('attributes', None).get('Caters', None)
                t_data['Coat Check'] = data.get('attributes', None).get('Coat Check', None)
                t_data['Corkage'] = data.get('attributes', None).get('Corkage', None)
                t_data['Delivery'] = data.get('attributes', None).get('Delivery', None)
                dr = data.get('attributes', None).get('Dietary Restrictions', None)
                if dr is not None:
                    t_data['res_dairy-free'] = dr.get('dairy-free')
                    t_data['res_gluten-free'] = dr.get('gluten-free')
                    t_data['res_vegan'] = dr.get('vegan')
                    t_data['res_kosher'] = dr.get('kosher')
                    t_data['res_halal'] = dr.get('halal')
                    t_data['res_soy-free'] = dr.get('soy-free')
                    t_data['res_vegetarian'] = dr.get('vegetarian')
                t_data['Dogs Allowed'] = data.get('attributes', None).get('Dogs Allowed', None)
                t_data['Drive-Thru'] = data.get('attributes', None).get('Drive-Thru', None)
                t_data['Good For Dancing'] = data.get('attributes', None).get('Good For Dancing', None)
                t_data['Good For Groups'] = data.get('attributes', None).get('Good For Groups', None)
                t_data['Good for Kids'] = data.get('attributes', None).get('Good for Kids', None)
                t_data['Good For Kids'] = data.get('attributes', None).get('Good For Kids', None)
                gf = data.get('attributes', None).get('Good For', None)
                if gf is not None:
                    t_data['goodfor_dessert'] = gf.get('dessert')
                    t_data['goodfor_latenight'] = gf.get('latenight')
                    t_data['goodfor_lunch'] = gf.get('lunch')
                    t_data['goodfor_dinner'] = gf.get('dinner')
                    t_data['goodfor_brunch'] = gf.get('brunch')
                    t_data['goodfor_breakfast'] = gf.get('breakfast')
                t_data['Happy Hour'] = data.get('attributes', None).get('Happy Hour', None)
                t_data['Has TV'] = data.get('attributes', None).get('Has TV', None)
                mu = data.get('attributes', None).get('Music', None)
                if mu is not None:
                    t_data['music_dj'] = mu.get('dj')
                    t_data['music_background_music'] = mu.get('background_music')
                    t_data['music_jukebox'] = mu.get('jukebox')
                    t_data['music_live'] = mu.get('live')
                    t_data['music_video'] = mu.get('video')
                    t_data['music_karaoke'] = mu.get('karaoke')
                    t_data['music_playlist'] = mu.get('playlist')
                t_data['Noise Level'] = data.get('attributes', None).get('Noise Level', None)
                t_data['Open 24 Hours'] = data.get('attributes', None).get('Open 24 Hours', None)
                t_data['Order at Counter'] = data.get('attributes', None).get('Order at Counter', None)
                t_data['Outdoor Seating'] = data.get('attributes', None).get('Outdoor Seating', None)
                pk = data.get('attributes', None).get('Parking', None)
                if pk is not None:
                    t_data['parking_garage'] = pk.get('garage')
                    t_data['parking_street'] = pk.get('street')
                    t_data['parking_validated'] = pk.get('validated')
                    t_data['parking_lot'] = pk.get('lot')
                    t_data['parking_valet'] = pk.get('valet')
                pt = data.get('attributes', None).get('Payment Types', None)
                if pt is not None:
                    t_data['pt_amex'] = pt.get('amex')
                    t_data['pt_cash_only'] = pt.get('cash_only')
                    t_data['pt_visa'] = pt.get('visa')
                    t_data['pt_mastercard'] = pt.get('mastercard')
                    t_data['pt_discover'] = pt.get('discover')
                t_data['Price Range'] = data.get('attributes', None).get('Price Range', None)
                t_data['Smoking'] = data.get('attributes', None).get('Smoking', None)
                t_data['Take-out'] = data.get('attributes', None).get('Take-out', None)
                t_data['Takes Reservations'] = data.get('attributes', None).get('Takes Reservations', None)
                t_data['Waiter Service'] = data.get('attributes', None).get('Waiter Service', None)
                t_data['Wheelchair Accessible'] = data.get('attributes', None).get('Wheelchair Accessible', None)
                t_data['Wi-Fi'] = data.get('attributes', None).get('Wi-Fi', None)
                # dataset.append(data.get('business_id'):t_data)
                dataset[data.get('business_id')] = t_data
        with open('../data/dataset_business_v2.json', 'w') as outfile:
            json.dump(dataset, outfile, indent=2)


def get_nominal_integer_dict(nominal_vals):
    d = {}
    for val in nominal_vals:
        if val not in d:
            current_max = max(d.values()) if len(d) > 0 else -1
            d[val] = current_max + 1
    return d


def convert_to_integer(srs):
    d = get_nominal_integer_dict(srs)
    return srs.map(lambda x: d[x])


def convert_strings_to_integer(df):
    ret = pd.DataFrame()
    for column_name in df:
        column = df[column_name]
        if column.dtype == 'string' or column.dtype == 'object':
            ret[column_name] = convert_to_integer(column)
        else:
            ret[column_name] = column
    return ret



def update_review():
    def merge_dict(x,y):
        z = x.copy()
        z.update(y)
        return z


    with open('../data/dataset_user_v2.json') as infile:
        user_data = json.loads(infile.read())
    with open('../data/dataset_business_v2.json') as infile:
        busi_data = json.loads(infile.read())

    reviews = []
    rate = []
    with open('../data/dataset_review.csv') as infile:
        for line in infile:
            u, b, r = line.strip().split(',')
            if user_data.get(u) is not None and busi_data.get(b) is not None:
                md = merge_dict(user_data[u], busi_data[b])
                md['y_rate'] = r
                reviews.append(md)
                rate.append(int(r))
    reviews = convert_strings_to_integer(pd.DataFrame(reviews))
    reviews.to_csv("../data/processed_reivews.csv")
    
    # print "working sklearn"

    # y = rate
    # # reviews = reviews.drop("y_rate")
    # X = reviews.values.tolist()

    # print reviews
    # data = pd.read_csv("../data/dataset_review_combined.csv")
    # print(data)
    # data = reviews
    # v = DictVectorizer(sparse=False)
    # X = v.fit_transform(reviews)
    # print v.get_feature_names()
    # print DictVectorizer().fit_transform(reviews).get_feature_names()


    # get x
    # get y
    # data.dropna()
    # y = dfList = np.asarray(data['y_rate'].tolist())
    # del data['y_rate']
    # x = reviews
    # x = DictVectorizer(sparse=False).fit_transform(reviews)
    
    # print x
    # x_missing = x.copy()
    # x_missing[np.where(missing)]
    # y = np.asarray(rate)

    # x = scipy.sparse.csr_matrix(x)
    # with open('x_all_data.dat', 'wb') as outfile:
    #     pickle.dump(x, outfile, pickle.HIGHEST_PROTOCOL)
    # with open('y_all_data.dat', 'wb') as outfile:
    #     pickle.dump(y, outfile, pickle.HIGHEST_PROTOCOL)

    # np.save(file('x_all_data.bin', 'wb'), X)
    # np.save(file('y_all_data.bin', 'wb'), y)
    # x.dump('x_all_data.bin')
    # y.dump('y_all_data.bin')

def run_model():

    # x1 = np.load(file("x_data.bin", 'rb'))
    # y1 = np.load(file("y_data.bin", 'rb'))
    # x2 = np.load(file("x2_data.bin", "rb"))
    # y2 = np.load(file("x2_data.bin", "rb"))
    # # x = np.concatenate((x1,x2), axis=0)
    # # y = np.concatenate((y1,y2), axis=0)
    # x = np.vstack([x1,x2])
    # y = np.vstack([y1,y2])
    # x = np.load(file("x_all_data.bin", 'rb'))
    # y = np.load(file("y_all_data.bin", 'rb'))
    # with open('x_all_data.dat', 'rb') as infile:
    #     x = pickle.load(infile)
    # with open('y_all_data.dat', 'rb') as infile:
    #     y = pickle.load(infile)

    reviews = pd.read_csv("../data/processed_reivews.csv")
    y = reviews['y_rate'].tolist()
    reviews = reviews.drop("y_rate", axis=1)
    x = reviews.values.tolist()

    print 'fit_transform finished'
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.1, random_state=0)
    print 'cross_validation finished'
    pipeline = Pipeline([
      ("Imputer", Imputer(missing_values='NaN',
                          strategy="mean",
                          axis=0)),
      ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
          # ('one_hot_encoding', OneHotEncoder(categorical_features=)),
          ('anova', SelectKBest(f_classif, k=10)),
          ('pca', decomposition.PCA(n_components=10))

        ]))])),
      # ('classifier', SVC())
      ('classifier', GaussianNB())
    ])
    """ """
    im = Imputer(missing_values='NaN',
                          strategy="mean",
                          axis=0)
    x = im.fit_transform(X_train)
    print x.shape
    # y = im.fit_transform(y_train)
    anova_k = SelectKBest(f_classif, k=10)
    anova_k.fit_transform(x, y_train)
    print anova_k.scores_
    print len(anova_k.scores_)
    cols_list = reviews.columns.values.tolist()
    list_scores = anova_k.scores_

    cols_scores = {}
    for i in range(len(cols_list)):
        if str(list_scores[i]) != 'nan':
            cols_scores[cols_list[i]] = list_scores[i]
        else:
            continue

    #Sort the score dictionary
    print '*** After sorted ****'
    sorted_cols = OrderedDict(sorted(cols_scores.items(), key=lambda t: t[1],reverse=True))
    print sorted_cols
    """ pipeline Results """
    # pipeline.fit(X_train, y_train)
    # joblib.dump(pipeline, "model.pkl")

    # print 'fit finished'
    # y_pred = pipeline.predict(X_test)
    # print 'predict finished'
    # print(precision_recall_fscore_support(y_test, y_pred))



    # Results = {}
    # precision = metrics.precision_score(y_test, y_pred)
    # recall = metrics.recall_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred)
    # accuracy = accuracy_score(y_test, y_pred)

    # data = {'precision':precision,
    # 'recall':recall,
    # 'f1_score':f1,
    # 'accuracy':accuracy}

    # Results['clf'] = data
    # cols = ['precision', 'recall', 'f1_score', 'accuracy']
    # print pd.DataFrame(Results).T[cols].T

    """ """
    # print pipeline.named_steps['features']
    # v = DictVectorizer(sparse=False)
    # X = v.fit_transform(reviews[201780:201790])
    # print X
    # pd.DataFrame(reviews).to_csv('../data/dataset_review_combined.csv')
    # 
def load_model():
    reviews = pd.read_csv("../data/processed_reivews.csv")
    y = reviews['y_rate'].tolist()
    reviews = reviews.drop("y_rate", axis=1)
    x = reviews.values.tolist()

    print 'fit_transform finished'
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.1, random_state=0)
    pipeline = joblib.load("model.pkl")
    y_pred = pipeline.predict(X_test)
    print 'predict finished'
    # print(precision_recall_fscore_support(y_test, y_pred))
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
 
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    # Results = {}
    # precision = metrics.precision_score(y_test, y_pred, average="micro")
    # recall = metrics.recall_score(y_test, y_pred, average="micro")
    # f1 = metrics.f1_score(y_test, y_pred, average="micro")
    # accuracy = accuracy_score(y_test, y_pred)

    # data = {'precision':precision,
    # 'recall':recall,
    # 'f1_score':f1,
    # 'accuracy':accuracy}

    # Results['clf'] = data
    # cols = ['precision', 'recall', 'f1_score', 'accuracy']
    # print pd.DataFrame(Results).T[cols].T


def main():
    # clean_user_data()
    # clean_busi_data()
    # update_review()
    run_model()
    # load_model()
    # busi_df = pd.read_json('../data/dataset_business_v2.json')
    # # print busi_df.transpose()
    # user_df = pd.read_json('../data/dataset_user_v2.json')
    # # print user_df.transpose()
    # review_df = pd.read_csv('../data/dataset_review.csv', header=None)
    # print review_df
if __name__ == '__main__':
    main()
