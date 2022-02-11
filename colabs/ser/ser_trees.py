import copy
import os
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (GridSearchCV, train_test_split)
from sklearn.tree import DecisionTreeClassifier

import ser_algo as ser


class SerDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, dec_tree_source: DecisionTreeClassifier, with_reduction=False):
        self.dec_tree_target = copy.deepcopy(dec_tree_source)
        self.with_reduction = with_reduction
    
    def fit(self, X_target, y_target):
        self.X_target = X_target
        self.y_target = y_target
        
        if self.with_reduction:
            ser.SER(0, self.dec_tree_target, X_target, y_target, original_ser=self.with_reduction)
        else:
            ser.SER(0, self.dec_tree_target, X_target, y_target,
                    original_ser=self.with_reduction,
                    no_red_on_cl=True,
                    cl_no_red=[0],
                    ext_cond=True)            
        return self

    def predict(self, X=None):
        return self.dec_tree_target.predict(X)
    
    def predict_proba(self, X):
        return self.dec_tree_target.predict_proba(X)


class SerRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rand_forest_source: RandomForestClassifier, bootstrap=False, with_reduction=False):
        self.rand_forest_target = rand_forest_source
        self.bootstrap = bootstrap
        self.with_reduction = with_reduction
    
    def fit(self, X_target, y_target):
        self.X_target = X_target
        self.y_target = y_target
        
        if self.with_reduction:
            self.rand_forest_target = ser.SER_RF(self.rand_forest_target, self.X_target, self.y_target,
                                                 original_ser=self.with_reduction)
        else:
            self.rand_forest_target = ser.SER_RF(self.rand_forest_target, self.X_target, self.y_target,
                                                 original_ser=self.with_reduction, no_red_on_cl=True,
                                                 cl_no_red=[1])
        return self

    def predict(self, X=None):
        return self.rand_forest_target.predict(X)
    
    def predict_proba(self, X):
        return self.rand_forest_target.predict_proba(X)


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
        

def transfer_learn_random_forest(rf_clf, with_reduction=False):
    rf_model_pkl = 'tl_RF_model_wr.pkl'
    if os.path.isfile(os.path.join(os.getcwd(), rf_model_pkl)):
        return joblib.load(rf_model_pkl)
    

def find_best_estimator(X, y, model_pkl='./models/RF_best_model.pkl'):
    rf_model_pkl = model_pkl
    if os.path.isfile(rf_model_pkl):
        return joblib.load(rf_model_pkl)
    
    rf = RandomForestClassifier()
    parameters = {
        'n_estimators': [5, 10, 15, 20],
        'max_depth': [2, 4, 8, 16, 32, None]
    }
    
    print(f"Shapes X_src: {X.shape}, y_src: {y.shape}")

    cv = GridSearchCV(rf, parameters, cv=5)
    cv.fit(X, y)

    print_results(cv)
    
    
    joblib.dump(cv.best_estimator_, rf_model_pkl)
    
    return cv.best_estimator_    


class TestSerTransferLearn(unittest.TestCase):
    RANDOM_FOREST_MODEL = os.path.join(os.getcwd(), 'models', 'random_forest_08202021.joblib')
    COUNT_VEC_TRANSFORM = os.path.join(os.getcwd(), 'models', 'count_vectorizer_08202021.joblib')
    
    @staticmethod
    def gen_synthetic_data():
        # Generate training source data
        np.random.seed(0)

        ns = 200
        ns_perclass = ns // 2
        mean_1 = (1, 1)
        var_1 = np.diag([1, 1])
        mean_2 = (3, 3)
        var_2 = np.diag([2, 2])
        X_src = np.r_[np.random.multivariate_normal(mean_1, var_1, size=ns_perclass),
                np.random.multivariate_normal(mean_2, var_2, size=ns_perclass)]
        y_src = np.zeros(ns)
        y_src[ns_perclass:] = 1
        
        # Generate training target data
        nt = 50
        # imbalanced
        nt_0 = nt // 10
        mean_1 = (6, 3)
        var_1 = np.diag([4, 1])
        mean_2 = (5, 5)
        var_2 = np.diag([1, 3])

        X_target = np.r_[np.random.multivariate_normal(mean_1, var_1, size=nt_0),
                np.random.multivariate_normal(mean_2, var_2, size=nt - nt_0)]
        y_target = np.zeros(nt)
        y_target[nt_0:] = 1

        # Generate testing target data
        nt_test = 1000
        nt_test_perclass = nt_test // 2
        X_target_test = np.r_[np.random.multivariate_normal(mean_1, var_1, size=nt_test_perclass),
                        np.random.multivariate_normal(mean_2, var_2, size=nt_test_perclass)]
        y_target_test = np.zeros(nt_test)
        y_target_test[nt_test_perclass:] = 1
        
        return X_src, y_src, X_target, y_target, X_target_test, y_target_test
    
    @staticmethod
    def get_digits_data():
        from sklearn.datasets import load_digits
        digits = load_digits()
        
        np.random.seed(0)
        
        X = digits.data[:200]
        y = (digits.target[:200] == 6).astype(int)

        X_tgt = digits.data[200:]
        y_tgt = (digits.target[200:] == 9).astype(int)
        
        X_src = X
        y_src = y
        
        # separating 5% & 95% of target data, stratified, random
        X_tgt_095, X_tgt_005, y_tgt_095, y_tgt_005 = train_test_split(
            X_tgt, y_tgt, test_size=0.05, stratify=y_tgt)

        return X_src, X_tgt_005, X_tgt_095, y_src, y_tgt_005, y_tgt_095
    
    @staticmethod
    def get_transfer_learned_rf_classifier(rf_clf, X_data, y_data, with_reduction=False):
        rf_model_pkl = 'tl_RF_model_no_red.pkl'
        if with_reduction:
            rf_model_pkl = 'tl_RF_model_red.pkl'
        
        if os.path.isfile(os.path.join(os.getcwd(), rf_model_pkl)):
            return joblib.load(rf_model_pkl)

        rf_clf_tgt = SerRandomForestClassifier(rf_clf, with_reduction=with_reduction)
        rf_clf_tgt.fit(X_data, y_data)
        
        joblib.dump(rf_clf_tgt, rf_model_pkl)
        
        return rf_clf_tgt
    
    @staticmethod
    def train_src_classifier(X_src, y_src):
        # Source classifier
        clf_source = DecisionTreeClassifier(max_depth=None)
        clf_source.fit(X_src, y_src)
        return clf_source
    
    @staticmethod
    def get_labeled_data():
        np.random.seed(0)
        
        N_FEATURES_SRC = 400
        N_FEATURES_TGT = 402
        

        persuasion_data = pd.read_csv('cleaned_persuasion_data.csv')
        persuasion_data['Unit'] = persuasion_data['Unit'].str.replace('donation','patch')
        persuasion_data['Unit'] = persuasion_data['Unit'].str.replace('charity','commit')
        persuasion_data['Unit'] = persuasion_data['Unit'].str.replace('donation','patches')
        
        id2label = dict(zip(persuasion_data['persuasion_id'], persuasion_data['er_label_1']))

        X = persuasion_data['Unit'][:N_FEATURES_SRC]
        y = persuasion_data['persuasion_id'].to_numpy()[:N_FEATURES_SRC]

        X_tgt = persuasion_data['Unit'][N_FEATURES_SRC:][:N_FEATURES_TGT]
        y_tgt = persuasion_data['persuasion_id'].to_numpy()[N_FEATURES_SRC:][:N_FEATURES_TGT]
        
        assert X_tgt.size == N_FEATURES_TGT
        assert y_tgt.size == N_FEATURES_TGT
        
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
        
        X_src = tfidf.fit_transform(X).toarray()
        y_src = y
        
        X_tgt = tfidf.fit_transform(X_tgt).toarray()
        y_tgt = y_tgt
        
        print(f"Shapes X_src: {X_src.shape}, y_src: {y_src.shape}")
        print(f"Shapes X_tgt: {X_tgt.shape}, y_tgt: {y_tgt.shape}")
        
        return X_src, y_src, X_tgt, y_tgt, X, y, tfidf, id2label
    
    
    @staticmethod
    def get_trained_random_forest(X_src, y_src):
        
        rf_model_pkl = 'models/RF_tuned_model.pkl'
        return find_best_estimator(X_src, y_src, model_pkl=rf_model_pkl)


    def test_src_classifier(self):
        Xs, ys, _, _, Xt_test, yt_test = TestSerTransferLearn.gen_synthetic_data()
        clf_src = TestSerTransferLearn.train_src_classifier(Xs, ys)
        score_src_src = clf_src.score(Xs, ys)
        score_src_tgt = clf_src.score(Xt_test, yt_test)
        
        print('Training score Source model: {:.3f}'.format(score_src_src))
        print('Testing score Source model: {:.3f}'.format(score_src_tgt))
        
        self.assertEqual(1.000, score_src_src)
        self.assertEqual(0.516, score_src_tgt)
        
    def test_tgt_classifier(self):
        Xs, ys, Xt, yt, Xt_test, yt_test = TestSerTransferLearn.gen_synthetic_data()
        clf_src = TestSerTransferLearn.train_src_classifier(Xs, ys)
        clf_tgt = SerDecisionTreeClassifier(clf_src, with_reduction=True)
        clf_tgt.fit(Xt, yt)
        score_tgt_tgt = clf_tgt.score(Xt_test, yt_test)
        print('Testing score transferred model ({}) : {:.3f}'.format("ser", score_tgt_tgt))
        self.assertEqual(0.647, score_tgt_tgt)
        
    def test_random_forest(self):
        MAX = 5
        N_EST = 3
        
        X_src, X_tgt_005, X_tgt_095, y_src, y_tgt_005, y_tgt_095 = TestSerTransferLearn.get_digits_data()
        
        rf_or = RandomForestClassifier(n_estimators=N_EST, max_depth=MAX)
        rf_or.fit(X_src, y_src)
        score = rf_or.score(X_tgt_005, y_tgt_005)
        self.assertEqual(0.8625, score)
     
        rf_clf_tgt = SerRandomForestClassifier(rf_or, with_reduction=True)
        rf_clf_tgt.fit(X_tgt_005, y_tgt_005)
        
        rf_clf_tgt_1 = SerRandomForestClassifier(rf_or)
        rf_clf_tgt_1.fit(X_tgt_005, y_tgt_005)
        
        score_ser = rf_clf_tgt.score(X_tgt_095, y_tgt_095)
        score_ser_no_red = rf_clf_tgt_1.score(X_tgt_095, y_tgt_095)
        
        print('score ser:', score_ser)
        print('score ser no red:', score_ser_no_red)

        self.assertEqual(0.8918918918918919, score_ser)
        self.assertEqual(0.8951878707976269, score_ser_no_red)
    
    def test_load_model(self):
        self.assertTrue(os.path.isfile(TestSerTransferLearn.RANDOM_FOREST_MODEL))
        self.assertTrue(os.path.isfile(TestSerTransferLearn.COUNT_VEC_TRANSFORM))

        rf_exp_1_clfr = joblib.load(TestSerTransferLearn.RANDOM_FOREST_MODEL)
        cv_exp_1_tfr = joblib.load(TestSerTransferLearn.COUNT_VEC_TRANSFORM)
        self.assertIsNotNone(rf_exp_1_clfr)
        self.assertIsNotNone(cv_exp_1_tfr)

        self.assertEqual('task-related-inquiry',
            rf_exp_1_clfr.predict(cv_exp_1_tfr.transform(["Are you involved with charities?"]))[0])
        self.assertEqual('logical-appeal',
            rf_exp_1_clfr.predict(cv_exp_1_tfr.transform(["These children really need assistance"]))[0])

    def test_transfer_learning(self):
        def expect_is_none(dataset_arr):
            for ds in dataset_arr:
                self.assertIsNotNone(ds)

        def expect_is_not_empty(dataset_arr):
            for ds in dataset_arr:
                self.assertTrue(ds.size > 0)

        X_src, y_src, X_tgt, y_tgt, _, _, tfidf_cv, _ = TestSerTransferLearn.get_labeled_data()
        rf_exp_1_clfr = TestSerTransferLearn.get_trained_random_forest(X_src, y_src)
        expect_is_none([rf_exp_1_clfr])

        X_tgt_095, X_tgt_005, y_tgt_095, y_tgt_005 = train_test_split(X_tgt, y_tgt, test_size=0.05, stratify=y_tgt)
        expect_is_none([X_tgt_095, X_tgt_005, y_tgt_095, y_tgt_005])
        expect_is_not_empty([X_tgt_095, X_tgt_005, y_tgt_095, y_tgt_005])

        score_src = rf_exp_1_clfr.score(X_tgt_095, y_tgt_095)
        print('score src:', score_src)
        y_pred = rf_exp_1_clfr.predict(tfidf_cv.transform(["Are you involved with charities?"]))
        self.assertEqual(1, y_pred[0])
        
        rf_clf_tgt_red = TestSerTransferLearn.get_transfer_learned_rf_classifier(rf_exp_1_clfr, X_tgt_005, y_tgt_005, with_reduction=True)
        score_ser = rf_clf_tgt_red.score(X_tgt_095, y_tgt_095)
        
        rf_clf_tgt_no_red = TestSerTransferLearn.get_transfer_learned_rf_classifier(rf_exp_1_clfr, X_tgt_005, y_tgt_005)
        score_ser_no_red = rf_clf_tgt_no_red.score(X_tgt_095, y_tgt_095)
        
        print('score ser:', score_ser)
        print('score ser no red:', score_ser_no_red)
        
    
    def test_persuasion_api(self):
        import api
        
        _, _, X_tgt, y_tgt, _, _, tfidf, id2label = TestSerTransferLearn.get_labeled_data()
        X_tgt_095, X_tgt_005, y_tgt_095, y_tgt_005 = train_test_split(X_tgt, y_tgt, test_size=0.05, stratify=y_tgt)
        base_predictor = api.PredictPersuasionStrategy('./models/RF_tuned_model.pkl', tfidf)
        self.assertIsNotNone(base_predictor)
        print('(base) score src:', base_predictor.model.score(X_tgt_095, y_tgt_095))
        print('(base) score src:', base_predictor.model.score(X_tgt_005, y_tgt_005))
        
        y_pred_base = base_predictor.get_label("Are you sure you are doing this?", mapping=id2label)['label']
        self.assertEqual('self-modeling', y_pred_base)
        
        no_red_predictor = api.PredictPersuasionStrategy('./models/tl_RF_model_no_red.pkl', tfidf)
        print('(no_red) score src:', no_red_predictor.model.score(X_tgt_095, y_tgt_095))
        print('(no_red) score src:', no_red_predictor.model.score(X_tgt_005, y_tgt_005))
        y_pred_no_red = no_red_predictor.get_label("Are you sure you are doing this?", mapping=id2label)['label']
        self.assertEqual('credibility-appeal', y_pred_no_red)

        red_predictor = api.PredictPersuasionStrategy('./models/tl_RF_model_red.pkl', tfidf)
        print('(red) score src:', red_predictor.model.score(X_tgt_095, y_tgt_095))
        print('(red) score src:', red_predictor.model.score(X_tgt_005, y_tgt_005))
        y_pred_red = red_predictor.get_label("Are you sure you are doing this?", mapping=id2label)['label']
        self.assertEqual(y_pred_no_red, y_pred_red)
        

if __name__ == '__main__':
    unittest.main()
