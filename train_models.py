import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
import pickle as pkl
from sklearn.metrics import matthews_corrcoef, make_scorer
import numpy as np
import sys
from sqlalchemy import create_engine
import datetime


def log(msg):
    '''Simple logging with timestamp.'''
    print(f'{datetime.datetime.now()} {msg}')



def main(symbol, dataset, filename, print_all_results=False, save_pickle=False, print_features_ranking=False):

    cols = dataset.columns
    target_col = [x for x in cols if 'threshold_is_met' in x]
    target_col = target_col[0]
    print(f'target_col: {target_col}')

    X = dataset.drop(target_col, axis=1)
    y = dataset[target_col]

    # Steps to run in sci-kit pipeline
    # PCA() is feature selection
    # SMOTE() artifically creates more instances where classification=1
    # Very important to have SMOTE after PCA, otherwise would be serious data leakage
    # SVC is the model (Support vector classification) - SVM but for classification
    steps = [('reduce_dim', PCA()),('over', SMOTETomek()), ('model', SVC())]
    pipeline = Pipeline(steps=steps)

    # Stratified K Fold, Much better than train-test-split
    # Divides data in 5 splits, trains on 4, then tests on 1
    # Repeats this process until each split has been used as test
    # Averages the results, Very unbiased way of testing models
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # Grid Search for different parameters
    # These can be expanded but it will take a lot of computer power
    param_grid = {
        'reduce_dim__n_components': [10, 25, 50],
        'model__kernel': ["rbf"],
        'model__gamma': [0.001],
        'model__C': [10, 100, 1000],
    }
    # Grid Search Over Parameters
    gs = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv, scoring=make_scorer(matthews_corrcoef), n_jobs=-1)
    gs.fit(X, y)
    # Various results from testing (Scores, parameters and run times)
    results = gs.cv_results_

    if print_all_results is True:
        for key, value in results.items():
            print(key, value)

    # Best score from any of the folds
    print(f'{symbol} MCC Score:', gs.best_score_, filename)
    # Best parameter set
    print(gs.best_params_)


    model = gs.best_estimator_
    # Refit the pca before saving
    pca = PCA(n_components=gs.best_params_["reduce_dim__n_components"]).fit(X)
    # number of components
    n_pcs= pca.components_.shape[0]
    # get the index of the most important feature on EACH component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

    initial_feature_names = X.columns
    # get the names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]


    if print_features_ranking is True:
        # LIST COMPREHENSION HERE AGAIN
        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

        # build the dataframe
        df = pd.DataFrame(dic.items())
        df.to_csv(f'{symbol} features used.csv', index=False)
        print(df.to_string())

    if save_pickle is True and gs.best_score_ > 0.35:
        if not os.path.exists('pickles'):
            os.makedirs('pickles')

        # Save best model
        with open(f'./pickles/{filename[:-4]}_MODEL.pkl', 'wb') as f:
            pkl.dump(model,f)

        # Save pca
        with open(f'./pickles/{filename[:-4]}_PCA.pkl', 'wb') as f:
            pkl.dump(pca,f)
        print('Pickles saved')

    return symbol, gs.best_score_, filename



def run_models(timeframe, rows, percent, csv_suffix, only_use_high_performance_models=True, folder=None):
    if not folder:
        raise ValueError(f'The function run_models needs to be passed a folder string value to find the dataset csvs for csv_suffix: {csv_suffix}')

    print(f'only_use_high_performance_models: {only_use_high_performance_models}\n')
    # print(f'folder function parameter value: {folder}')

    symbols = ['SPY', 'QQQ', 'SPXL']

    percent = str(percent).replace('.', 'point')
    values_for_new_dataframe = []
    for x, symbol in enumerate(symbols):
        try:
            log(f'Processing {symbol}   {x} of {len(symbols)}')
            temp_list = []
            path = f'../../trade_analysis/output_csvs{folder}'
            filename = f'dataset_{symbol}_{timeframe}_label_close_{rows}_rows_{percent}_percent_threshold_is_met_{csv_suffix}.csv'
            # print(f'Dataset path: {path}')
            # print(f'Dataset filename: {filename}')
            print(f'Dataset: {path}{filename}')

            if only_use_high_performance_models is True:
                high_performance_signal = get_high_performance_signals_from_sql(filename)
            else:
                high_performance_signal = False

            if only_use_high_performance_models is True and high_performance_signal is False:
                continue

            dataset = pd.read_csv(f'{path}{filename}')
            # print(f'Dataset path: {path}{filename}')
            # print(f'Dataset cols: {dataset.columns}')
            print(f'Dataset rows: {len(dataset)}')
            print(f'Dataset columns: {len(dataset.columns)}')
            print(f'Symbol: {symbol}')
            print(f'Timeframe: {timeframe}')

            symbol_to_add, best_score_to_add, filename_to_add = main(symbol, dataset, filename, print_all_results=False,
                                                                     save_pickle=True, print_features_ranking=False)
            temp_list.append(symbol_to_add)
            temp_list.append(best_score_to_add)
            temp_list.append(filename_to_add)
            values_for_new_dataframe.append(temp_list)
        except FileNotFoundError as e:
            print(e)
            print(f'\nERROR for {symbol} above\n\n')
            continue

        if not os.path.exists('csvs_to_combine'):
            os.makedirs('csvs_to_combine')

        df = pd.DataFrame(values_for_new_dataframe, columns=['symbol', 'performance', 'file_name'])
        df.to_csv(f'./csvs_to_combine/{symbol}_{timeframe}_{rows}_rows_{percent}_percent_results_{csv_suffix}.csv', index=False)
        print(f'{symbol}_{timeframe}_{rows}_rows_{percent}_percent_results_{csv_suffix}.csv successfully created\n')

    return df



def run_values_analytics():
    df = get_high_performance_signals_from_sql(filename=None, for_analytics=True)
    df = df[df.timeframe == 'daily']
    x = df["file_name"].tolist()

    new_symbols_list = ['PTON', 'NET', 'ZS', 'ZM', 'SNAP', 'BL', 'EVBG', 'U', 'CRSR', ' PINS']
    index_symbols = ['SPY', 'QQQ', 'SPXL']

    new_symbol_occurances = 0
    technicals_ocurrances = 0
    qqq_occurances = 0
    spy_occurances = 0
    spxl_occurances = 0
    five_rows_occurances = 0
    ten_rows_occurances = 0
    twenty_rows_occurances = 0
    thirty_rows_occurances = 0
    forty_rows_occurances = 0
    internals_only_occurances = 0
    internals_and_abstracted_occurances = 0
    abstracted_only_occurances = 0

    for value in x:

        for new_symbol in new_symbols_list:
            if new_symbol in value:
                new_symbol_occurances += 1

        if 'Technicals' in value:
            technicals_ocurrances += 1

        # for index_symbol in index_symbols:
        #     if index_symbol in value:
        #         index_occurances += 1

        if 'SPY' in value:
            spy_occurances += 1
            # print(value)
        if 'QQQ' in value:
            qqq_occurances += 1
        if 'SPXL' in value:
            spxl_occurances += 1
        if '5_rows' in value:
            five_rows_occurances += 1
        if '10_rows' in value:
            ten_rows_occurances += 1
        if '20_rows' in value:
            twenty_rows_occurances += 1
        if '30_rows' in value:
            thirty_rows_occurances += 1
        if '40_rows' in value:
            forty_rows_occurances += 1
        if 'Internals_Features' in value:
            internals_only_occurances += 1
        if 'Abstracted_Features' in value:
            abstracted_only_occurances += 1
        if 'Combined_Features' in value:
            internals_and_abstracted_occurances += 1

    print(f'New symbol occurances: {new_symbol_occurances} of {len(x)}   {new_symbol_occurances / len(x)}')
    print(f'Technicals datasets occurances:  {technicals_ocurrances} of {len(x)}   {technicals_ocurrances / len(x)}')
    print(f'SPY symbol occurances:  {spy_occurances} of {len(x)}   {spy_occurances / len(x)}')
    print(f'QQQ symbol occurances:  {qqq_occurances} of {len(x)}   {qqq_occurances / len(x)}')
    print(f'SPXL occurances:  {spxl_occurances} of {len(x)}   {spxl_occurances / len(x)}')
    print(f'five_rows_occurances:  {five_rows_occurances} of {len(x)}   {five_rows_occurances / len(x)}')
    print(f'ten_rows_occurances:  {ten_rows_occurances} of {len(x)}   {ten_rows_occurances / len(x)}')
    print(f'twenty_rows_occurances: {twenty_rows_occurances} of {len(x)}   {twenty_rows_occurances / len(x)}')
    print(f'thirty_rows_occurances:  {thirty_rows_occurances} of {len(x)}   {thirty_rows_occurances / len(x)}')
    print(f'forty_rows_occurances:  {forty_rows_occurances} of {len(x)}   {forty_rows_occurances / len(x)}')
    print(f'internals_only_occurances:  {internals_only_occurances} of {len(x)}   {internals_only_occurances / len(x)}')
    print(f'abstracted_only_occurances:  {abstracted_only_occurances} of {len(x)}   {abstracted_only_occurances / len(x)}')
    print(f'internals_and_abstracted_occurances:  {internals_and_abstracted_occurances} of {len(x)}   {internals_and_abstracted_occurances / len(x)}')



if __name__ == '__main__':
    # run_values_analytics()
    # exit()

    # try:

    row_periods_to_process = daily_rows_threshold

    csv_suffixes_and_target_folders = [
        ('Combined_Internals_and_Abstracted_Features', '/labels_with_conditions/combined_internals_and_abstracted_csvs/'),
    ]

    df_all = pd.DataFrame(columns=['symbol', 'performance', 'file_name'])
    for (csv_suffix, target_folder) in csv_suffixes_and_target_folders:
        for (period, threshold) in row_periods_to_process:
            df_to_add = run_models(timeframe, rows=period, percent=threshold, csv_suffix=csv_suffix, only_use_high_performance_models=False, folder=target_folder)



