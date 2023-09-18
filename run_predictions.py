from imblearn.pipeline import Pipeline
import pickle as pkl
import pandas as pd


def make_prediction(filename, this_is_being_run_for_production=True):
    # ----------------------------------------------------
    # New prediction, setup pipeline again but with the trained objects
    # You could also just manually enter the best parameters that came from the model training
    # e.g instead of ('reduce_dim', pca)
    # ('reduce_dim', PCA(n_components=100))
    with open(f'./pickles/{filename}_MODEL.pkl', 'rb') as f:
        model = pkl.load(f)

    with open(f'./pickles/{filename}_PCA.pkl', 'rb') as f:
        pca = pkl.load(f)

    steps = [('model', model)]
    pipeline = Pipeline(steps=steps)

    dataset = pd.read_csv(f'../../trade_analysis/output_csvs/{filename}.csv')

    # Delete the "threshold_is_met" Classification first column that is produced in preprocessing and used in model training (but not prediction)
    del dataset[dataset.columns[0]]

    if 'date_or_datetime' in dataset.columns:
        date_or_datetime_is_passed = True
        date_or_datetime = dataset['date_or_datetime'].iloc[-1]
        del dataset['date_or_datetime']
    else:
        date_or_datetime_is_passed = False

    if this_is_being_run_for_production is True:
        dataset = dataset.iloc[-1]

    dataset_orig = dataset.copy()
    pca.transform(dataset)

    predictions = pipeline.predict(dataset)
    dataset_orig["Prediction"] = predictions
    # dataset_orig.to_csv("predictions.csv", mode='a', index=False)

    if date_or_datetime_is_passed is True:
        return dataset_orig["Prediction"], date_or_datetime
    else:
        return dataset_orig["Prediction"]


if __name__ == '__main__':
    make_prediction()
