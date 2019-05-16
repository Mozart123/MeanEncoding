import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

# I created this function to make sure we are not overfitting. We can't overfit here by accident,
# because we don't have any access to validation labels. You must call this function for each fold in CV.
# I wrote this code for a competition, so there are also other things not related to mean encoding.

def process_fold(train_fold_x, train_fold_y, valid_fold_x, X_test):
    #You can do whatever you want in this function and you won't overfit in CV results. => No bad surprise in submissions.
    #The reason for that is we don't have access to validation data target. Overfitting to CV results from making decisions based on
    #Validation set targets.
    
    #Things you can do here:
    # - Target encoding
    # - Binning
    # - Oversampling (preprocessing fold data is faster for that)
    # - Other FE
    
    #This didn't improve CV score, but you won't get false hopes by overfitting to CV.
    cols_to_bin = np.random.choice(train_fold_x.columns, 10) #I chose these randomly, just to show that function works
    cols_to_encode = cols_to_bin
    
    print('Binning {} columns. (Replace)...'.format(len(cols_to_bin)))
    for col in cols_to_bin:
        est = KBinsDiscretizer(n_bins=25, encode='ordinal', strategy='quantile') #Can try different things
        est.fit(train_fold_x[col].values.reshape((-1,1)))
        # You may also fit to pd.concat([train_fold_x, valid_fold_x, X_test]) but I'm not sure which one works better

        train_fold_x[col] = est.transform(train_fold_x[col].values.reshape((-1,1)))
        valid_fold_x[col] = est.transform(valid_fold_x[col].values.reshape((-1,1)))
        X_test[col] = est.transform(X_test[col].values.reshape((-1,1)))
    
    #Cascaded mean encoding (This part is a little bit crappy, but I didn't have time to fix)
    #By encoding all columns this way, you can obtain a feature with 0.89 auc on its own without a model. But it didn't contribute to overall CV.
    
    print('Target encoding {} columns. (Add columns)...'.format(len(cols_to_encode)))
    num_valid = len(valid_fold_x)
    for col in cols_to_encode:
        train_fold_x['target'] = train_fold_y
        train_encoded, test_encoded = mean_encode(train_fold_x, pd.concat([valid_fold_x, X_test], axis = 0), [col], 'target', reg_method='k_fold',
                alpha=1, add_random=False, rmean=0, rstd=0.1, folds=4)
        train_fold_x.drop('target', axis = 1, inplace = True)
        
        train_encoded.drop('index', axis = 1, inplace = True)
        test_encoded.drop('index', axis = 1, inplace = True)
        
        train_fold_x.reset_index(drop = True, inplace = True)
        valid_fold_x.reset_index(drop = True, inplace = True)
        X_test.reset_index(drop = True, inplace = True)
        
        valid_encoded = test_encoded.iloc[:num_valid].reset_index(drop = True)
        test_encoded = test_encoded.iloc[num_valid:].reset_index(drop = True)
    
        train_fold_x = pd.concat([train_encoded, train_fold_x], axis = 1).reset_index(drop = True)
        valid_fold_x =  pd.concat([valid_encoded, valid_fold_x], axis = 1).reset_index(drop = True)
        X_test =  pd.concat([test_encoded, X_test], axis = 1).reset_index(drop = True)
    
    print('Fold processing done.')
    #Goes back into training
    return [train_fold_x, train_fold_y, valid_fold_x, X_test]


#Target encoding function, this is from: https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study
def mean_encode(train_data, test_data, columns, target_col, reg_method=None,
                alpha=0, add_random=False, rmean=0, rstd=0.1, folds=1):
    length_train = len(train_data)
    '''Returns a DataFrame with encoded columns'''
    encoded_cols = []
    target_mean_global = train_data[target_col].mean()
    for col in columns:
        # Getting means for test data
        nrows_cat = train_data.groupby(col)[target_col].count()
        target_means_cats = train_data.groupby(col)[target_col].mean()
        target_means_cats_adj = (target_means_cats*nrows_cat + 
                                 target_mean_global*alpha)/(nrows_cat+alpha)
        # Mapping means to test data
        encoded_col_test = test_data[col].map(target_means_cats_adj)
        # Getting a train encodings
        if reg_method == 'expanding_mean':
            train_data_shuffled = train_data.sample(frac=1, random_state=1)
            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]
            cumcnt = train_data_shuffled.groupby(col).cumcount()
            encoded_col_train = cumsum/(cumcnt)
            encoded_col_train.fillna(target_mean_global, inplace=True)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))
        elif (reg_method == 'k_fold') and (folds > 1):
            kfold = StratifiedKFold(n_splits = folds, shuffle=True, random_state=1).split(train_data[target_col].values, train_data[target_col])
            parts = []
            for tr_in, val_ind in kfold:
                                # divide data
                    
                
                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]
                # getting means on data for estimation (all folds except estimated)
                nrows_cat = df_for_estimation.groupby(col)[target_col].count()
                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
                target_means_cats_adj = (target_means_cats*nrows_cat + 
                                         target_mean_global*alpha)/(nrows_cat+alpha)
                # Mapping means to estimated fold
                encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
                if add_random:
                    encoded_col_train_part = encoded_col_train_part + normal(loc=rmean, scale=rstd, 
                                                                             size=(encoded_col_train_part.shape[0]))
                # Saving estimated encodings for a fold
                parts.append(encoded_col_train_part)
            encoded_col_train = pd.concat(parts, axis=0)
            encoded_col_train.fillna(target_mean_global, inplace=True)
        else:
            encoded_col_train = train_data[col].map(target_means_cats_adj)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))

        # Saving the column with means
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = target_mean_global
        encoded_cols.append(pd.DataFrame({'mean_'+target_col+'_'+col:encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    #Modified to reindex
    all_encoded = all_encoded.reset_index()
    return (all_encoded.iloc[:length_train].reset_index(drop = True), 
            all_encoded.iloc[length_train:].reset_index(drop = True)
           )
