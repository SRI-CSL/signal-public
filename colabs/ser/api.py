import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from textsplit.tools import SimpleSentenceTokenizer

DEFAULT_PARAM_GRID = {
    'n_estimators': [5, 10, 15, 20],
    'max_depth': [2, 4, 8, 16, 32, None]
}

def load_model(pickle_filepath):
    obj = None
    if pickle_filepath is None or not os.path.isfile(pickle_filepath):
        print(f"WARN: {os.path.abspath(pickle_filepath)} not found")
        return obj, False

    obj = joblib.load(pickle_filepath)

    return obj, True


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


def get_trained_random_forest(X_src, y_src, param_grid, cv=5, debug=False):
    
    rf_model_pkl = 'RF_tuned_model.pkl'
    rf, success = load_model(rf_model_pkl)
    if not success:
        return rf
    
    rf = RandomForestClassifier()
    if not param_grid:
        raise ValueError("param grid is None")

    cv = GridSearchCV(rf, param_grid, cv=cv)
    cv.fit(X_src, y_src)

    if debug:
        print(f"Shapes X_src: {X_src.shape}, y_src: {y_src.shape}")
        print_results(cv)

    return cv.best_estimator_


class PersuasionModel(object):
    def __init__(self, path_to_model=None, vectorizer=None):
        if not isinstance(path_to_model, str):
            raise ValueError("Expected a file path/to/model.pkl")

        self.requires_training = True
        model, success = load_model(path_to_model)
        if not success:
            raise ValueError("Unable to load serialized model")
        
        self.trained_model = model
        
        # general vectorizer
        self.vectorizer = vectorizer
        

    def _fit(self, X):
        if not self.vectorizer:
            raise ValueError("vectorizer is None")
        # fits a TfIdf vectorizer to the X
        self.vectorizer.fit(X)
        return self

    def _transform(self, X):
        if not self.vectorizer:
            raise ValueError("vectorizer is None")
        return self.vectorizer.transform(X)

    def fit_transform_vectorizer(self, X):
        if not self.vectorizer:
            raise ValueError("vectorizer is None")
        return self.vectorizer.fit_transform(X).toarray()

    def predict(self, X_input):
        # Returns the predicted class in an array
        # TODO re-implement this to match the API used by the persuasion model
        # The following code assumes we are using the MultinomialNB classifier.
        y_pred = self.trained_model.predict(X_input)
        return y_pred

    def score(self, X, y):
        return self.trained_model.score(X, y)

    def predict_prob(self, X):
         return self.trained_model.predict_proba(X)

    def to_pkl(self, model_file_path="./best_model_state_er.pkl"):
        # Saves the trained classifier for future use.
        joblib.dump(self.trained_model, model_file_path)


class PredictPersuasionStrategy(object):
    def __init__(self, model, vectorizer):
        if isinstance(model, str) and os.path.isfile(model):
            self.model = PersuasionModel(model, vectorizer)
        else:
            self.model = model

    def fit_vectorizer(self, X):
        if not self.model:
            raise ValueError("Model is not available")
        
        self.model.fit_transform_vectorizer(X)
        return self

    def get_label(self, txt_segment, mapping=None):
        # vectorize the text segments and make a prediction
        ts_vector = self.model._transform([txt_segment])
        prediction = self.model.predict(ts_vector)
        # round the predict prob value and set to new variable
        prob_pred = self.model.predict_prob(ts_vector)[0]
        confidence = round(max(prob_pred), 3)
        output = {'label': prediction[0], 'confidence': confidence}
        if mapping:
            output = {'label': mapping.get(prediction[0], 'UNKNOWN'), 'confidence': confidence}

        return output

    def get_prediction(self, email_body_text, split_by_par=False):
        if split_by_par:
            # split email body text into paragraphs?
            txt_segment_array = email_body_text.split('\n\n')
        else:
            # by sentence
            sentence_tokenizer = SimpleSentenceTokenizer()
            txt_segment_array = sentence_tokenizer(email_body_text)
        return [self.get_label(text_segment) for text_segment in txt_segment_array]


if __name__ == '__main__':
    pass
