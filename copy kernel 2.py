import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import warnings
from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
from lightgbm import LGBMClassifier
from scipy.stats import ranksums
import random
from xgboost import XGBClassifier
warnings.simplefilter(action='ignore', category=FutureWarning)
from bayes_opt import BayesianOptimization

dtypes = pd.read_csv(r'D:\kaggle\home credit risk\copy kernel generate table\final 1400 features dtypes.csv')
dtypes = dtypes.set_index('col_name')
dtype_dict = {i: dtypes.loc[i][0] for i in dtypes.index}
df = pd.read_csv(r'D:\kaggle\home credit risk\copy kernel generate table\final 1400 features.csv', dtype=dtype_dict)
null = (df.isnull().sum()/ df.shape[0]).sort_values()
hist_size_0_1 =hist_feature_selection_0_1(df).sort_values()
hist_size_train_test = hist_feature_selection_train_test(df).sort_values()
merge = list(set(hist_size_0_1[:1000].index) & set(hist_size_train_test[-1000:].index) & set(null[null>0.6].index))

def reduce_mem_usage(data, verbose=True):
    start_mem = data.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))

    for col in data.columns:
        col_type = data[col].dtype

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return data


def one_hot_encoder(data, nan_as_category=True):
    original_columns = list(data.columns)
    categorical_columns = [col for col in data.columns \
                           if not pd.api.types.is_numeric_dtype(data[col].dtype)]
    for c in categorical_columns:
        if nan_as_category:
            data[c].fillna('NaN', inplace=True)
        values = list(data[c].unique())
        for v in values:
            data[str(c) + '_' + str(v)] = (data[c] == v).astype(np.uint8)
    data.drop(categorical_columns, axis=1, inplace=True)
    return data, [c for c in data.columns if c not in original_columns]


file_path = r'D:\kaggle\home credit risk\DATA\\'


def application_train_test(file_path=file_path, nan_as_category=True):
    def application_housing_pca(df,num):

        housing_list = [i for i in df.columns for j in ['AVG', 'MEDI', 'MODE'] if j in i]
        doc_list = [i for i in df.columns if 'FLAG_DOCUMENT' in i]

        housing_list.append('REGION_POPULATION_RELATIVE')
        df_housing = df[housing_list].copy()
        for i in housing_list:
            if df_housing[i].dtype == 'float64':
                df_housing[i].fillna(df_housing[i].mean(), inplace=True)
            else:
                df_housing[i][df_housing[i].isnull()] = 'Unknown'
                df_housing[i] = df_housing[i].astype('category')
                df_housing[i] = df_housing[i].cat.codes
        # housing
        pca = PCA(n_components=num)
        housing_pca = pca.fit_transform(df_housing)

        df_housing_pca = pd.DataFrame(housing_pca, columns=['Housing PCA '+str(i) for i in range(num)])
        df = df.join(df_housing_pca)
        df.drop(housing_list, inplace=True, axis=1)
        # doc
        #pca = PCA(n_components=num)
        #doc_pca = pca.fit_transform(df[doc_list])

        #df_doc_pca = pd.DataFrame(doc_pca, columns=['Doc PCA '+str(i) for i in range(num)])
        #df = df.join(df_doc_pca)

        #df.drop(doc_list, inplace=True, axis=1)
        del df_housing_pca, housing_pca, df_housing
        gc.collect()
        return (df)
    # Read data and merge
    df_train = pd.read_csv(file_path + 'application_train.csv')
    df_test = pd.read_csv(file_path + 'application_test.csv')
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    del df_train, df_test
    gc.collect()

    # Remove some rows with values not present in test set
    df.drop(df[df['CODE_GENDER'] == 'XNA'].index, inplace=True)
    df.drop(df[df['NAME_INCOME_TYPE'] == 'Maternity leave'].index, inplace=True)
    df.drop(df[df['NAME_FAMILY_STATUS'] == 'Unknown'].index, inplace=True)

    # Remove some empty features
    df.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
             'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
             'FLAG_DOCUMENT_21'], axis=1, inplace=True)

    # Replace some outliers
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df.loc[df['OWN_CAR_AGE'] > 80, 'OWN_CAR_AGE'] = np.nan
    df.loc[df['REGION_RATING_CLIENT_W_CITY'] < 0, 'REGION_RATING_CLIENT_W_CITY'] = np.nan
    df.loc[df['AMT_INCOME_TOTAL'] > 1e8, 'AMT_INCOME_TOTAL'] = np.nan
    df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT'] > 10, 'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 40, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    df, _ = one_hot_encoder(df, nan_as_category)

    # Some new features
    df['app missing'] = df.isnull().sum(axis=1).values

    df['app EXT_SOURCE mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['app EXT_SOURCE std'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['app EXT_SOURCE prod'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_1 * EXT_SOURCE_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['app EXT_SOURCE_1 * EXT_SOURCE_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_2 * EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_1 * DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_2 * DAYS_EMPLOYED'] = df['EXT_SOURCE_2'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_3 * DAYS_EMPLOYED'] = df['EXT_SOURCE_3'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_1 / DAYS_BIRTH'] = df['EXT_SOURCE_1'] / df['DAYS_BIRTH']
    df['app EXT_SOURCE_2 / DAYS_BIRTH'] = df['EXT_SOURCE_2'] / df['DAYS_BIRTH']
    df['app EXT_SOURCE_3 / DAYS_BIRTH'] = df['EXT_SOURCE_3'] / df['DAYS_BIRTH']

    df['app AMT_CREDIT - AMT_GOODS_PRICE'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['app AMT_CREDIT / AMT_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['app AMT_CREDIT / AMT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['app AMT_CREDIT / AMT_INCOME_TOTAL'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    df['app AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = df['AMT_INCOME_TOTAL'] / 12. - df['AMT_ANNUITY']
    df['app AMT_INCOME_TOTAL / AMT_ANNUITY'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
    df['app AMT_INCOME_TOTAL - AMT_GOODS_PRICE'] = df['AMT_INCOME_TOTAL'] - df['AMT_GOODS_PRICE']
    df['app AMT_INCOME_TOTAL / CNT_FAM_MEMBERS'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['app AMT_INCOME_TOTAL / CNT_CHILDREN'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    df['app most popular AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] \
        .isin([225000, 450000, 675000, 900000]).map({True: 1, False: 0})
    df['app popular AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] \
        .isin([1125000, 1350000, 1575000, 1800000, 2250000]).map({True: 1, False: 0})

    df['app OWN_CAR_AGE / DAYS_BIRTH'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['app OWN_CAR_AGE / DAYS_EMPLOYED'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']

    df['app DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['app DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['app DAYS_EMPLOYED - DAYS_BIRTH'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
    df['app DAYS_EMPLOYED / DAYS_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    df['app CNT_CHILDREN / CNT_FAM_MEMBERS'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    df = application_housing_pca(df,3)

    return reduce_mem_usage(df)


def bureau_and_balance(file_path=file_path, nan_as_category=True):
    df_bureau_b = reduce_mem_usage(pd.read_csv(file_path + 'bureau_balance.csv'), verbose=False)

    # Some new features in bureau_balance set
    tmp = df_bureau_b[['SK_ID_BUREAU', 'STATUS']].groupby('SK_ID_BUREAU')
    tmp_last = tmp.last()
    tmp_last.columns = ['First_status']
    df_bureau_b = df_bureau_b.join(tmp_last, how='left', on='SK_ID_BUREAU')
    tmp_first = tmp.first()
    tmp_first.columns = ['Last_status']
    df_bureau_b = df_bureau_b.join(tmp_first, how='left', on='SK_ID_BUREAU')
    del tmp, tmp_first, tmp_last
    gc.collect()

    tmp = df_bureau_b[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').last()
    tmp = tmp.apply(abs)
    tmp.columns = ['Month']
    df_bureau_b = df_bureau_b.join(tmp, how='left', on='SK_ID_BUREAU')
    del tmp
    gc.collect()

    tmp = df_bureau_b.loc[df_bureau_b['STATUS'] == 'C', ['SK_ID_BUREAU', 'MONTHS_BALANCE']] \
        .groupby('SK_ID_BUREAU').last()
    tmp = tmp.apply(abs)
    tmp.columns = ['When_closed']
    df_bureau_b = df_bureau_b.join(tmp, how='left', on='SK_ID_BUREAU')
    del tmp
    gc.collect()

    df_bureau_b['Month_closed_to_end'] = df_bureau_b['Month'] - df_bureau_b['When_closed']

    for c in range(6):
        tmp = df_bureau_b.loc[df_bureau_b['STATUS'] == str(c), ['SK_ID_BUREAU', 'MONTHS_BALANCE']] \
            .groupby('SK_ID_BUREAU').count()
        tmp.columns = ['DPD_' + str(c) + '_cnt']
        df_bureau_b = df_bureau_b.join(tmp, how='left', on='SK_ID_BUREAU')
        df_bureau_b['DPD_' + str(c) + ' / Month'] = df_bureau_b['DPD_' + str(c) + '_cnt'] / df_bureau_b['Month']
        del tmp
        gc.collect()
    df_bureau_b['Non_zero_DPD_cnt'] = df_bureau_b[
        ['DPD_1_cnt', 'DPD_2_cnt', 'DPD_3_cnt', 'DPD_4_cnt', 'DPD_5_cnt']].sum(axis=1)

    df_bureau_b, bureau_b_cat = one_hot_encoder(df_bureau_b, nan_as_category)

    # Bureau balance: Perform aggregations
    aggregations = {}
    for col in df_bureau_b.columns:
        aggregations[col] = ['mean','sum'] if col in bureau_b_cat else ['min', 'max', 'size']
    df_bureau_b_agg = df_bureau_b.groupby('SK_ID_BUREAU').agg(aggregations)
    df_bureau_b_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_bureau_b_agg.columns.tolist()])
    del df_bureau_b
    gc.collect()

    df_bureau = reduce_mem_usage(pd.read_csv(file_path + 'bureau.csv'), verbose=False)

    # Replace\remove some outliers in bureau set

    # fill na
    df_bureau.loc[df_bureau['CREDIT_ACTIVE'] == 'Closed', ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT']] = \
        df_bureau[df_bureau['CREDIT_ACTIVE'] == 'Closed'][['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT']].fillna(0)

    # credit sum = credit sum limit + credit sum debt
    df_bureau.loc[df_bureau['CREDIT_ACTIVE'] == 'Active', 'AMT_CREDIT_SUM_LIMIT'] = \
        df_bureau[df_bureau['CREDIT_ACTIVE'] == 'Active']['AMT_CREDIT_SUM'] - \
        df_bureau[df_bureau['CREDIT_ACTIVE'] == 'Active']['AMT_CREDIT_SUM_DEBT']

    df_bureau.loc[df_bureau['AMT_ANNUITY'] > .8e8, 'AMT_ANNUITY'] = np.nan
    df_bureau.loc[df_bureau['AMT_CREDIT_SUM'] > 3e8, 'AMT_CREDIT_SUM'] = np.nan
    df_bureau.loc[df_bureau['AMT_CREDIT_SUM_DEBT'] > 1e8, 'AMT_CREDIT_SUM_DEBT'] = np.nan
    df_bureau.loc[df_bureau['AMT_CREDIT_MAX_OVERDUE'] > .8e8, 'AMT_CREDIT_MAX_OVERDUE'] = np.nan
    df_bureau.loc[df_bureau['DAYS_ENDDATE_FACT'] < -10000, 'DAYS_ENDDATE_FACT'] = np.nan
    df_bureau.loc[(df_bureau['DAYS_CREDIT_UPDATE'] > 0) | (
            df_bureau['DAYS_CREDIT_UPDATE'] < -40000), 'DAYS_CREDIT_UPDATE'] = np.nan
    df_bureau.loc[df_bureau['DAYS_CREDIT_ENDDATE'] < -10000, 'DAYS_CREDIT_ENDDATE'] = np.nan

    df_bureau.drop(df_bureau[df_bureau['DAYS_ENDDATE_FACT'] < df_bureau['DAYS_CREDIT']].index, inplace=True)
    df_bureau.drop('CREDIT_CURRENCY',axis=1,inplace=True)


    # Some new features in bureau set
    df_bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_DEBT'] = df_bureau['AMT_CREDIT_SUM'] - df_bureau[
        'AMT_CREDIT_SUM_DEBT']
    df_bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_LIMIT'] = df_bureau['AMT_CREDIT_SUM'] - df_bureau[
        'AMT_CREDIT_SUM_LIMIT']
    df_bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_OVERDUE'] = df_bureau['AMT_CREDIT_SUM'] - df_bureau[
        'AMT_CREDIT_SUM_OVERDUE']

    df_bureau['bureau DAYS_CREDIT - CREDIT_DAY_OVERDUE'] = df_bureau['DAYS_CREDIT'] - df_bureau['CREDIT_DAY_OVERDUE']
    df_bureau['bureau DAYS_CREDIT - DAYS_CREDIT_ENDDATE'] = df_bureau['DAYS_CREDIT'] - df_bureau['DAYS_CREDIT_ENDDATE']
    df_bureau['bureau DAYS_CREDIT - DAYS_ENDDATE_FACT'] = df_bureau['DAYS_CREDIT'] - df_bureau['DAYS_ENDDATE_FACT']
    df_bureau['bureau DAYS_CREDIT_ENDDATE - DAYS_ENDDATE_FACT'] = df_bureau['DAYS_CREDIT_ENDDATE'] - df_bureau[
        'DAYS_ENDDATE_FACT']
    df_bureau['bureau DAYS_CREDIT_UPDATE - DAYS_CREDIT_ENDDATE'] = df_bureau['DAYS_CREDIT_UPDATE'] - df_bureau[
        'DAYS_CREDIT_ENDDATE']

    df_bureau['FLAG_overdue'] = df_bureau['AMT_CREDIT_SUM_OVERDUE'].apply(lambda x: 1 if x > 0 else 0)

    # replace high correlation column and low variance column

    # Categorical features with One-Hot encode
    df_bureau['CREDIT_TYPE'] = df_bureau['CREDIT_TYPE'].apply(
        lambda x: x if x in ['Consumer credit', 'Credit card'] else 'other')
    df_bureau['CREDIT_ACTIVE'] = df_bureau['CREDIT_ACTIVE'].apply(
        lambda x: x if x in ['Closed', 'Active'] else 'other')
    df_bureau, bureau_cat = one_hot_encoder(df_bureau, nan_as_category)

    # Bureau balance: merge with bureau.csv
    df_bureau = df_bureau.join(df_bureau_b_agg, how='left', on='SK_ID_BUREAU')
    df_bureau.drop('SK_ID_BUREAU', axis=1, inplace=True)
    del df_bureau_b_agg
    gc.collect()

    # Bureau and bureau_balance aggregations for application set
    categorical = bureau_cat + bureau_b_cat
    aggregations = {}
    for col in df_bureau.columns:
        aggregations[col] = ['mean','sum'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(aggregations)
    df_bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in df_bureau_agg.columns.tolist()])

    # Bureau: Active credits
    active_agg = df_bureau[df_bureau['CREDIT_ACTIVE_Active'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    active_agg.columns = pd.Index(['BURO_ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    df_bureau_agg = df_bureau_agg.join(active_agg, how='left')
    del active_agg
    gc.collect()

    # Bureau: Closed credits
    closed_agg = df_bureau[df_bureau['CREDIT_ACTIVE_Closed'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    closed_agg.columns = pd.Index(['BURO_CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    df_bureau_agg = df_bureau_agg.join(closed_agg, how='left')

    # bureau: amt annuity ==0
    annuity_0_agg = df_bureau[df_bureau['AMT_ANNUITY'].isnull()].groupby('SK_ID_CURR').agg(aggregations)
    annuity_0_agg.columns = pd.Index(['BURO_annuity_0_' + e[0] + "_" + e[1].upper() for e in annuity_0_agg.columns.tolist()])
    df_bureau_agg = df_bureau_agg.join(annuity_0_agg, how='left')

    # bureau: amt annuity >0
    annuity_non_0_agg = df_bureau[df_bureau['AMT_ANNUITY']>0].groupby('SK_ID_CURR').agg(aggregations)
    annuity_non_0_agg.columns = pd.Index(
        ['BURO_annuity_non_0_' + e[0] + "_" + e[1].upper() for e in annuity_non_0_agg.columns.tolist()])
    df_bureau_agg = df_bureau_agg.join(annuity_non_0_agg, how='left')

    del closed_agg, df_bureau
    gc.collect()

    return reduce_mem_usage(df_bureau_agg)


def previous_application(file_path=file_path, nan_as_category=True):
    def goods_cat(x):
        if x in ['XNA', 'Other']:
            return 'XNA'
        elif x in ['Mobile', 'Consumer Electronics', 'Computers', 'Photo / Cinema Equipment',
                   'Clothing and Accessories', 'Jewelry', 'Sport and Leisure', 'Tourism',
                   'Fitness', 'Additional Service', 'Weapon', 'Animals', 'Direct Sales']:
            return 'electronics & leisure'
        else:
            return 'home & car & edu & medi'

    df_prev = pd.read_csv(file_path + 'previous_application.csv')

    # Replace some outliers
    df_prev.loc[df_prev['AMT_CREDIT'] > 6000000, 'AMT_CREDIT'] = np.nan
    df_prev.loc[df_prev['SELLERPLACE_AREA'] > 3500000, 'SELLERPLACE_AREA'] = np.nan
    df_prev[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
             'DAYS_LAST_DUE', 'DAYS_TERMINATION']].replace(365243, np.nan, inplace=True)

    # category
    df_prev.drop('WEEKDAY_APPR_PROCESS_START', axis=1, inplace=True)

    df_prev['NAME_SELLER_INDUSTRY'] = df_prev['NAME_SELLER_INDUSTRY'].apply(lambda x:
                                                                            'other' if x not in ['XNA',
                                                                                                 'Consumer electronics',
                                                                                                 'Connectivity']
                                                                            else x)
    df_prev['CHANNEL_TYPE'] = df_prev['CHANNEL_TYPE'].apply(lambda x:
                                                            'other' if x not in ['Credit and cash offices',
                                                                                 'Country-wide'] else x)
    df_prev['NAME_PORTFOLIO'] = df_prev['NAME_PORTFOLIO'].apply(lambda x:
                                                                'other' if x not in ['POS', 'Cash'] else x)

    df_prev['NAME_GOODS_CATEGORY'] = df_prev['NAME_GOODS_CATEGORY'].apply(lambda x: goods_cat(x))
    df_prev['NAME_TYPE_SUITE'] = df_prev['NAME_TYPE_SUITE'].apply(lambda x: 'other' if x != 'Unaccompanied' else x)
    df_prev['CODE_REJECT_REASON'] = df_prev['CODE_REJECT_REASON'].apply(lambda x:
                                                                        'other' if x not in ['XAP', 'HC'] else x)
    df_prev['NAME_PAYMENT_TYPE'] = df_prev['NAME_PAYMENT_TYPE'].apply(lambda x:
                                                                      'other' if x not in [
                                                                          'Cash through the bank'] else x)
    df_prev['NAME_CASH_LOAN_PURPOSE'] = df_prev['NAME_CASH_LOAN_PURPOSE'].apply(lambda x:
                                                                                'other' if x not in ['XAP',
                                                                                                     'XNA'] else x)

    # Some new features
    df_prev['prev missing'] = df_prev.isnull().sum(axis=1).values
    df_prev['prev AMT_APPLICATION / AMT_CREDIT'] = df_prev['AMT_APPLICATION'] / df_prev['AMT_CREDIT']
    df_prev['prev AMT_APPLICATION - AMT_CREDIT'] = df_prev['AMT_APPLICATION'] - df_prev['AMT_CREDIT']

    df_prev['prev AMT_APPLICATION - AMT_GOODS_PRICE'] = df_prev['AMT_APPLICATION'] - df_prev['AMT_GOODS_PRICE']
    df_prev['prev AMT_GOODS_PRICE - AMT_CREDIT'] = df_prev['AMT_GOODS_PRICE'] - df_prev['AMT_CREDIT']
    df_prev['prev DAYS_FIRST_DRAWING - DAYS_FIRST_DUE'] = df_prev['DAYS_FIRST_DRAWING'] - df_prev['DAYS_FIRST_DUE']
    df_prev['prev DAYS_TERMINATION less -500'] = (df_prev['DAYS_TERMINATION'] < -500).astype(int)
    df_prev['DAYS_LAST_DUE - DAYS_TERMINATION'] = df_prev['DAYS_LAST_DUE'] - df_prev['DAYS_TERMINATION']
    #df_prev = df_prev.drop(['AMT_APPLICATION', 'AMT_GOODS_PRICE', 'DAYS_TERMINATION'], axis=1)
    df_prev['avg loan terms'] = df_prev['AMT_CREDIT'] / df_prev['AMT_ANNUITY']


    # Categorical features with One-Hot encode
    df_prev, categorical = one_hot_encoder(df_prev, nan_as_category)

    # Aggregations for application set
    aggregations = {}
    for col in df_prev.columns:
        aggregations[col] = ['mean','sum'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_prev_agg = df_prev.groupby('SK_ID_CURR').agg(aggregations)
    df_prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in df_prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications
    approved_agg = df_prev[df_prev['NAME_CONTRACT_STATUS_Approved'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    approved_agg.columns = pd.Index(['PREV_APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    df_prev_agg = df_prev_agg.join(approved_agg, how='left')
    del approved_agg
    gc.collect()

    # Previous Applications: Refused Applications
    refused_agg = df_prev[df_prev['NAME_CONTRACT_STATUS_Refused'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    refused_agg.columns = pd.Index(['PREV_REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    df_prev_agg = df_prev_agg.join(refused_agg, how='left')
    del refused_agg
    gc.collect()

    # cash loans application
    cash_loan_agg = df_prev[df_prev['NAME_CONTRACT_TYPE_Cash loans'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    cash_loan_agg.columns = pd.Index(
        ['PREV_Cash loans_' + e[0] + "_" + e[1].upper() for e in cash_loan_agg.columns.tolist()])
    df_prev_agg = df_prev_agg.join(cash_loan_agg, how='left')

    del cash_loan_agg
    gc.collect()
    # consumer loans
    consumer_loan_agg = df_prev[df_prev['NAME_CONTRACT_TYPE_Consumer loans'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    consumer_loan_agg.columns = pd.Index(
        ['PREV_Consumer loans_' + e[0] + "_" + e[1].upper() for e in consumer_loan_agg.columns.tolist()])
    df_prev_agg = df_prev_agg.join(consumer_loan_agg, how='left')
    del consumer_loan_agg
    gc.collect()

    # Revolving loans
    Revolving_loan_agg = df_prev[df_prev['NAME_CONTRACT_TYPE_Revolving loans'] == 1].groupby('SK_ID_CURR').agg(
        aggregations)
    Revolving_loan_agg.columns = pd.Index(
        ['PREV_Revolving loans_' + e[0] + "_" + e[1].upper() for e in Revolving_loan_agg.columns.tolist()])
    df_prev_agg = df_prev_agg.join(Revolving_loan_agg, how='left')

    del Revolving_loan_agg
    gc.collect()


    del df_prev
    gc.collect()

    return reduce_mem_usage(df_prev_agg)


def pos_cash(file_path=file_path, nan_as_category=True):
    df_pos = pd.read_csv(file_path + 'POS_CASH_balance.csv')

    # Replace some outliers
    df_pos.loc[df_pos['CNT_INSTALMENT_FUTURE'] > 60, 'CNT_INSTALMENT_FUTURE'] = np.nan

    # Some new features
    df_pos['FLAG_DPD'] = df_pos['SK_DPD'].apply(lambda x: 1 if x>0 else 0)

    # Categorical features with One-Hot encode
    df_pos, categorical = one_hot_encoder(df_pos, nan_as_category)

    # Aggregations for application set
    aggregations = {}
    for col in df_pos.columns:
        aggregations[col] = ['mean','sum'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_pos_agg = df_pos.groupby('SK_ID_CURR').agg(aggregations)
    df_pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in df_pos_agg.columns.tolist()])

    # Count POS lines
    df_pos_agg['POS_COUNT'] = df_pos.groupby('SK_ID_CURR').size()

    df_ins = pd.read_csv(file_path + 'installments_payments.csv')
    df_ins.loc[df_ins['NUM_INSTALMENT_VERSION'] > 70, 'NUM_INSTALMENT_VERSION'] = np.nan
    df_ins.loc[df_ins['DAYS_ENTRY_PAYMENT'] < -4000, 'DAYS_ENTRY_PAYMENT'] = np.nan
    total_ins = df_ins.groupby('SK_ID_PREV').sum()[['AMT_INSTALMENT', 'AMT_PAYMENT']]
    pos_ins = df_pos[['SK_ID_PREV', 'SK_ID_CURR']].groupby('SK_ID_PREV').last().join(total_ins).groupby('SK_ID_CURR').sum()
    pos_ins.columns=['POS_total_installment','POS_total_payment']
    df_pos_agg = df_pos_agg.join(pos_ins)

    # number of install version


    del df_pos,
    gc.collect()

    return reduce_mem_usage(df_pos_agg)


def installments_payments(file_path=file_path, nan_as_category=True):
    df_ins = pd.read_csv(file_path + 'installments_payments.csv')

    # Replace some outliers
    df_ins.loc[df_ins['NUM_INSTALMENT_VERSION'] > 70, 'NUM_INSTALMENT_VERSION'] = np.nan
    df_ins.loc[df_ins['DAYS_ENTRY_PAYMENT'] < -4000, 'DAYS_ENTRY_PAYMENT'] = np.nan

    # Some new features
    df_ins['ins DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT'] = df_ins['DAYS_ENTRY_PAYMENT'] - df_ins['DAYS_INSTALMENT']
    df_ins['ins NUM_INSTALMENT_NUMBER_100'] = (df_ins['NUM_INSTALMENT_NUMBER'] == 100).astype(int)
    df_ins['ins DAYS_INSTALMENT more NUM_INSTALMENT_NUMBER'] = (
            df_ins['DAYS_INSTALMENT'] > df_ins['NUM_INSTALMENT_NUMBER'] * 50 / 3 - 11500 / 3).astype(int)
    df_ins['ins AMT_INSTALMENT - AMT_PAYMENT'] = df_ins['AMT_INSTALMENT'] - df_ins['AMT_PAYMENT']
    df_ins['ins AMT_PAYMENT / AMT_INSTALMENT'] = df_ins['AMT_PAYMENT'] / df_ins['AMT_INSTALMENT']

    # Categorical features with One-Hot encode
    df_ins, categorical = one_hot_encoder(df_ins, nan_as_category)

    # Aggregations for application set
    aggregations = {}
    for col in df_ins.columns:
        aggregations[col] = ['mean','sum'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_ins_agg = df_ins.groupby('SK_ID_CURR').agg(aggregations)
    df_ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in df_ins_agg.columns.tolist()])

    ins_ver = df_ins[['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION']].groupby('SK_ID_PREV').max()
    ins_ver.columns= ['SK_ID_CURR', 'INS_installment_ver_sum']
    ins_ver_sum = ins_ver.groupby('SK_ID_CURR').sum()
    ins_ver.columns = ['SK_ID_CURR', 'INS_installment_ver_mean']
    ins_ver_mean = ins_ver.groupby('SK_ID_CURR').mean()

    df_ins_agg = df_ins_agg.join(ins_ver_sum)
    df_ins_agg = df_ins_agg.join(ins_ver_mean)


    # Count installments lines
    df_ins_agg['INSTAL_COUNT'] = df_ins.groupby('SK_ID_CURR').size()
    del df_ins, ins_ver_sum, ins_ver, ins_ver_mean
    gc.collect()

    return reduce_mem_usage(df_ins_agg)


def credit_card_balance(file_path=file_path, nan_as_category=True):
    df_card = pd.read_csv(file_path + 'credit_card_balance.csv')

    # Replace some outliers
    df_card.loc[df_card['AMT_PAYMENT_CURRENT'] > 4000000, 'AMT_PAYMENT_CURRENT'] = np.nan
    df_card.loc[df_card['AMT_CREDIT_LIMIT_ACTUAL'] > 1000000, 'AMT_CREDIT_LIMIT_ACTUAL'] = np.nan

    # Some new features
    df_card['card missing'] = df_card.isnull().sum(axis=1).values
    df_card['card SK_DPD - MONTHS_BALANCE'] = df_card['SK_DPD'] - df_card['MONTHS_BALANCE']
    df_card['card SK_DPD_DEF - MONTHS_BALANCE'] = df_card['SK_DPD_DEF'] - df_card['MONTHS_BALANCE']
    df_card['card SK_DPD - SK_DPD_DEF'] = df_card['SK_DPD'] - df_card['SK_DPD_DEF']

    df_card['card AMT_TOTAL_RECEIVABLE - AMT_RECIVABLE'] = df_card['AMT_TOTAL_RECEIVABLE'] - df_card['AMT_RECIVABLE']
    df_card['card AMT_TOTAL_RECEIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = df_card['AMT_TOTAL_RECEIVABLE'] - df_card[
        'AMT_RECEIVABLE_PRINCIPAL']
    df_card['card AMT_RECIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = df_card['AMT_RECIVABLE'] - df_card[
        'AMT_RECEIVABLE_PRINCIPAL']

    df_card['card AMT_BALANCE - AMT_RECIVABLE'] = df_card['AMT_BALANCE'] - df_card['AMT_RECIVABLE']
    df_card['card AMT_BALANCE - AMT_RECEIVABLE_PRINCIPAL'] = df_card['AMT_BALANCE'] - df_card[
        'AMT_RECEIVABLE_PRINCIPAL']
    df_card['card AMT_BALANCE - AMT_TOTAL_RECEIVABLE'] = df_card['AMT_BALANCE'] - df_card['AMT_TOTAL_RECEIVABLE']

    df_card['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_ATM_CURRENT'] = df_card['AMT_DRAWINGS_CURRENT'] - df_card[
        'AMT_DRAWINGS_ATM_CURRENT']
    df_card['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_OTHER_CURRENT'] = df_card['AMT_DRAWINGS_CURRENT'] - df_card[
        'AMT_DRAWINGS_OTHER_CURRENT']
    df_card['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_POS_CURRENT'] = df_card['AMT_DRAWINGS_CURRENT'] - df_card[
        'AMT_DRAWINGS_POS_CURRENT']

    df_card['AMT_PAYMENT_CURRENT - AMT_PAYMENT_TOTAL_CURRENT'] = df_card['AMT_PAYMENT_CURRENT'] - df_card['AMT_PAYMENT_TOTAL_CURRENT']

    df_card['SK_DPD * AMT OBERDUE'] = df_card['SK_DPD'] * (df_card['AMT_INST_MIN_REGULARITY'] - df_card['AMT_PAYMENT_CURRENT'])

    df_card['available credit'] = df_card['AMT_CREDIT_LIMIT_ACTUAL'] - df_card['AMT_BALANCE']

    df_card = df_card.sort_values(by= ['SK_ID_PREV','MONTHS_BALANCE'])









    # Categorical features with One-Hot encode
    df_card, categorical = one_hot_encoder(df_card, nan_as_category)

    # Aggregations for application set
    aggregations = {}
    for col in df_card.columns:
        aggregations[col] = ['mean','sum'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_card_agg = df_card.groupby('SK_ID_CURR').agg(aggregations)
    df_card_agg.columns = pd.Index(['CARD_total_' + e[0] + "_" + e[1].upper() for e in df_card_agg.columns.tolist()])

    df_card_agg['CARD_total avg  DRAWING'] = df_card_agg['CARD_total_AMT_DRAWINGS_CURRENT_SUM']/ df_card_agg['CARD_total_CNT_DRAWINGS_CURRENT_SUM']
    df_card_agg['CARD_total avg  OTHER DRAWING'] = df_card_agg['CARD_total_AMT_DRAWINGS_OTHER_CURRENT_SUM'] / df_card_agg['CARD_total_CNT_DRAWINGS_OTHER_CURRENT_SUM']
    df_card_agg['CARD_total avg  ATM DRAWING'] = df_card_agg['CARD_total_AMT_DRAWINGS_ATM_CURRENT_SUM'] / df_card_agg['CARD_total_CNT_DRAWINGS_ATM_CURRENT_SUM']
    df_card_agg['CARD_total avg  POS DRAWING'] = df_card_agg['CARD_total_AMT_DRAWINGS_POS_CURRENT_SUM'] / df_card_agg['CARD_total_CNT_DRAWINGS_POS_CURRENT_SUM']


    # aggregations when credit card is used amt drawing >0
    aggregations = {}
    for col in df_card.columns:
        aggregations[col] = ['mean','sum'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_card_used_agg = df_card[df_card['AMT_DRAWINGS_ATM_CURRENT'] > 0].groupby('SK_ID_CURR').agg(aggregations)
    df_card_used_agg.columns = pd.Index(['CARD_used_' + e[0] + "_" + e[1].upper() for e in df_card_used_agg.columns.tolist()])

    df_card_used_agg['CARD_used avg  DRAWING'] = df_card_used_agg['CARD_used_AMT_DRAWINGS_CURRENT_SUM']/ df_card_used_agg['CARD_used_CNT_DRAWINGS_CURRENT_SUM']
    df_card_used_agg['CARD_used avg  OTHER DRAWING'] = df_card_used_agg['CARD_used_AMT_DRAWINGS_OTHER_CURRENT_SUM'] / df_card_used_agg['CARD_used_CNT_DRAWINGS_OTHER_CURRENT_SUM']
    df_card_used_agg['CARD_used avg  ATM DRAWING'] = df_card_used_agg['CARD_used_AMT_DRAWINGS_ATM_CURRENT_SUM'] / df_card_used_agg['CARD_used_CNT_DRAWINGS_ATM_CURRENT_SUM']
    df_card_used_agg['CARD_used avg  POS DRAWING'] = df_card_used_agg['CARD_used_AMT_DRAWINGS_POS_CURRENT_SUM'] / df_card_used_agg['CARD_used_CNT_DRAWINGS_POS_CURRENT_SUM']

    df_card_agg=df_card_agg.join(df_card_used_agg)



    # Count credit card lines
    df_card_agg['CARD_COUNT'] = df_card.groupby('SK_ID_CURR').size()
    df_card_agg['CARD_USED_COUNT']= df_card[df_card['AMT_DRAWINGS_ATM_CURRENT'].notnull()].groupby('SK_ID_CURR').size()

    # total balance
    latest_balance = df_card[['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','AMT_BALANCE']].groupby('SK_ID_PREV').last()
    latest_balance.columns = ['SK_ID_CURR', 'MONTHS_BALANCE', 'CARD_total balance']
    total_latest_balance = latest_balance.groupby('SK_ID_CURR').sum()['CARD_total balance']
    df_card_agg = df_card_agg.join(total_latest_balance)

    del df_card, latest_balance, total_latest_balance
    gc.collect()

    return reduce_mem_usage(df_card_agg)


def aggregate(file_path=file_path):
    warnings.simplefilter(action='ignore')

    print('-' * 20)
    print('1: application train & test (', time.ctime(), ')')
    print('-' * 20)
    df = application_train_test(file_path)
    print('     DF shape:', df.shape)

    print('-' * 20)
    print('2: bureau & balance (', time.ctime(), ')')
    print('-' * 20)
    bureau = bureau_and_balance(file_path)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    print('     DF shape:', df.shape)
    del bureau
    gc.collect()

    print('-' * 20)
    print('3: previous_application (', time.ctime(), ')')
    print('-' * 20)
    prev = previous_application(file_path)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    print('     DF shape:', df.shape)
    del prev
    gc.collect()

    print('-' * 20)
    print('4: POS_CASH_balance (', time.ctime(), ')')
    print('-' * 20)
    pos = pos_cash(file_path)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    print('     DF shape:', df.shape)
    del pos
    gc.collect()

    print('-' * 20)
    print('5: installments_payments (', time.ctime(), ')')
    print('-' * 20)
    ins = installments_payments(file_path)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    print('     DF shape:', df.shape)
    del ins
    gc.collect()

    print('-' * 20)
    print('6: credit_card_balance (', time.ctime(), ')')
    print('-' * 20)
    cc = credit_card_balance(file_path)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    print('     DF shape:', df.shape)
    del cc
    gc.collect()

    print('-' * 20)
    print('7: final dataset (', time.ctime(), ')')
    print('-' * 20)

    df['future terms'] = df['BURO_ACTIVE_DAYS_CREDIT_ENDDATE_SUM'] / 30 + df['POS_CNT_INSTALMENT_FUTURE_SUM'] + df[
        'AMT_CREDIT'] / df['AMT_ANNUITY']

    df['future payment'] = df['POS_total_installment'] - df['POS_total_payment'] + df['CARD_total balance'] + df[
        'AMT_CREDIT'] + df['BURO_AMT_CREDIT_SUM_DEBT_SUM']
    df['future payment/ income total'] = df['future payment'] / df['AMT_INCOME_TOTAL']
    df['future payment / CNT FAM MEMBERS'] = df['future payment'] / df['CNT_FAM_MEMBERS']
    df['future payment / CNT_CHILDREN'] = df['future payment'] / df['CNT_CHILDREN']
    df['future payment/ days birth'] = df['future payment'] / df['DAYS_BIRTH']

    df['avg future payment per term'] = df['future payment'] / df['future terms']

    return reduce_mem_usage(df)


# Kaggle has not engough memory to clean this dataset
# Aggregated dataset has 3411 featuresl


def clean_data(data):
    def corr_feature_with_target(feature, target):
        c0 = feature[target == 0].dropna()
        c1 = feature[target == 1].dropna()

        if set(feature.unique()) == set([0, 1]):
            diff = abs(c0.mean(axis=0) - c1.mean(axis=0))
        else:
            diff = abs(c0.median(axis=0) - c1.median(axis=0))

        p = ranksums(c0, c1)[1] if ((len(c0) >= 20) & (len(c1) >= 20)) else 2

        return [diff, p]
    warnings.simplefilter(action='ignore')

    # Removing empty features
    nun = data.nunique()
    empty = list(nun[nun <= 1].index)

    data.drop(empty, axis=1, inplace=True)
    print('After removing empty features there are {0:d} features'.format(data.shape[1]))

    # Removing features with the same distribution on 0 and 1 classes
    corr = pd.DataFrame(index=['diff', 'p'])
    ind = data[data['TARGET'].notnull()].index

    for c in data.columns.drop('TARGET'):
        corr[c] = corr_feature_with_target(data.loc[ind, c], data.loc[ind, 'TARGET'])
        gc.collect()

    corr = corr.T
    corr['diff_norm'] = abs(corr['diff'] / data.mean(axis=0))

    to_del_1 = corr[((corr['diff'] == 0) & (corr['p'] > .05))].index
    to_del_2 = corr[((corr['diff_norm'] < .5) & (corr['p'] > .05))].index
    to_del = list(set(to_del_1) & set(to_del_2))
    if 'SK_ID_CURR' in to_del:
        to_del.remove('SK_ID_CURR')

    data.drop(to_del, axis=1, inplace=True)
    print('After removing features with the same distribution on 0 and 1 classes there are {0:d} features'.format(
        data.shape[1]))
    del to_del, to_del_1, to_del_2
    gc.collect()

    # Removing features with not the same distribution on train and test datasets
    corr_test = pd.DataFrame(index=['diff', 'p'])
    target = data['TARGET'].notnull().astype(int)

    for c in data.columns.drop('TARGET'):
        corr_test[c] = corr_feature_with_target(data[c], target)

    corr_test = corr_test.T
    corr_test['diff_norm'] = abs(corr_test['diff'] / data.mean(axis=0))

    bad_features = corr_test[((corr_test['p'] < .05) & (corr_test['diff_norm'] > 1))].index
    bad_features = corr.loc[bad_features][corr['diff_norm'] == 0].index

    data.drop(bad_features, axis=1, inplace=True)
    print(
        'After removing features with not the same distribution on train and test datasets there are {0:d} features'.format(
            data.shape[1]))

    del corr, corr_test
    gc.collect()

    # Removing features not interesting for classifier
    clf = LGBMClassifier(random_state=0)
    train_index = data[data['TARGET'].notnull()].index
    train_columns = data.drop('TARGET', axis=1).columns

    score = 1
    new_columns = []
    while score > .6:
        train_columns = train_columns.drop(new_columns)
        clf.fit(data.loc[train_index, train_columns], data.loc[train_index, 'TARGET'])
        f_imp = pd.Series(clf.feature_importances_, index=train_columns)
        score = roc_auc_score(data.loc[train_index, 'TARGET'],
                              clf.predict_proba(data.loc[train_index, train_columns])[:, 1])
        new_columns = f_imp[f_imp > 0].index

    data.drop(train_columns, axis=1, inplace=True)
    print('After removing features not interesting for classifier there are {0:d} features'.format(data.shape[1]))

    return data


lgbm_params = {
    # 'nthread': 4,
    'n_estimators': 10000,
    'learning_rate': .05,
    'num_leaves': 12,
    'colsample_bytree': .7,
    'subsample': .7,
    'max_depth': 4,
    'reg_alpha': .07,
    'reg_lambda': .04,
    'min_split_gain': .0222415,
    'min_child_weight': 20,
    'verbose': -1,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'scale_pos_weight': 1,
    'n_jobs': 4,
    'is_unbalance':True
    'seed':1

}


def display_folds_importances(feature_importance_df_, n_folds=5):
    n_columns = 3
    n_rows = (n_folds + 1) // n_columns
    _, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 8 * n_rows))
    for i in range(n_folds):
        sns.barplot(x=i, y='index', data=feature_importance_df_.reset_index().sort_values(i, ascending=False).head(20),
                    ax=axes[i // n_columns, i % n_columns])
    sns.barplot(x='mean', y='index',
                data=feature_importance_df_.reset_index().sort_values('mean', ascending=False).head(20),
                ax=axes[n_rows - 1, n_columns - 1])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()


def cv_scores(df, num_folds, params, stratified=False, verbose=-1,
              save_train_prediction=False, train_prediction_file_name='train_prediction.csv',
              save_test_prediction=False, test_prediction_file_name='test_prediction.csv'):
    warnings.simplefilter('ignore')

    clf = LGBMClassifier(**params)

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    # Create arrays and dataframes to store results
    train_pred = np.zeros(train_df.shape[0])
    train_pred_proba = np.zeros(train_df.shape[0])

    test_pred = np.zeros(train_df.shape[0])
    test_pred_proba = np.zeros(train_df.shape[0])

    prediction = np.zeros(test_df.shape[0])

    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    df_feature_importance = pd.DataFrame(index=feats)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        print('Fold', n_fold, 'started at', time.ctime())
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc',
                verbose=verbose, early_stopping_rounds=200)

        train_pred[train_idx] = clf.predict(train_x, num_iteration=clf.best_iteration_)
        train_pred_proba[train_idx] = clf.predict_proba(train_x, num_iteration=clf.best_iteration_)[:, 1]
        test_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
        test_pred_proba[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        prediction += \
            clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        df_feature_importance[n_fold] = pd.Series(clf.feature_importances_, index=feats)

        print('Fold %2d AUC : %.6f' % (n_fold, roc_auc_score(valid_y, test_pred_proba[valid_idx])))
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    roc_auc_train = roc_auc_score(train_df['TARGET'], train_pred_proba)
    precision_train = precision_score(train_df['TARGET'], train_pred, average=None)
    recall_train = recall_score(train_df['TARGET'], train_pred, average=None)

    roc_auc_test = roc_auc_score(train_df['TARGET'], test_pred_proba)
    precision_test = precision_score(train_df['TARGET'], test_pred, average=None)
    recall_test = recall_score(train_df['TARGET'], test_pred, average=None)

    print('Full AUC score %.6f' % roc_auc_test)

    df_feature_importance.fillna(0, inplace=True)
    df_feature_importance['mean'] = df_feature_importance.mean(axis=1)

    # Write prediction files
    if save_train_prediction:
        df_prediction = train_df[['SK_ID_CURR', 'TARGET']]
        df_prediction['Prediction'] = test_pred_proba
        df_prediction.to_csv(train_prediction_file_name, index=False)
        del df_prediction
        gc.collect()

    if save_test_prediction:
        df_prediction = test_df[['SK_ID_CURR']]
        df_prediction['TARGET'] = prediction
        df_prediction.to_csv(test_prediction_file_name, index=False)
        del df_prediction
        gc.collect()

    return df_feature_importance, \
           [roc_auc_train, roc_auc_test,
            precision_train[0], precision_test[0], precision_train[1], precision_test[1],
            recall_train[0], recall_test[0], recall_train[1], recall_test[1], 0]



feature_importance, scor= cv_scores(df, 5, lgbm_params)


def xgboost_cv_scores(df, num_folds, params, stratified=False, verbose=-1, hard_sample=None, unsample_num=None,
                      save_train_prediction=False, train_prediction_file_name='train_prediction.csv',
                      save_test_prediction=True, test_prediction_file_name='test_prediction.csv'):
    warnings.simplefilter('ignore')
    warnings.filterwarnings('ignore')

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    default_row_index = list(train_df[train_df.TARGET == 1].index)
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                # random_state=1001,
                                )
    else:
        folds = KFold(n_splits=num_folds, shuffle=True,
                      # random_state=1001
                      )

    # Create arrays and dataframes to store results
    train_pred = np.zeros(train_df.shape[0])
    train_pred_proba = np.zeros(train_df.shape[0])

    test_pred = np.zeros(train_df.shape[0])
    test_pred_proba = np.zeros(train_df.shape[0])
    train_df_pred_proba = np.zeros(train_df.shape[0])

    prediction = np.zeros(test_df.shape[0])

    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    df_feature_importance = pd.DataFrame(index=feats)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        # print('Fold', n_fold, 'started at', time.ctime())
        clf = XGBClassifier(**params)
        train_idx = list(train_idx)
        if unsample_num:
            train_idx_0 = list(set(train_idx) - set(default_row_index))
            train_idx_0_delete = train_idx_0[:unsample_num]
            train_idx = list(set(train_idx) - set(train_idx_0_delete))
        default_index_train = list(set(default_row_index) - set(valid_idx))
        random.shuffle(default_index_train)
        train_idx.extend(default_index_train)
        if hard_sample:
            train_idx = list(set(train_idx) - set(hard_sample))

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc',
                verbose=verbose, early_stopping_rounds=50)

        train_pred[train_idx] = clf.predict(train_x)
        train_pred_proba[train_idx] = clf.predict_proba(train_x)[:, 1]
        test_pred[valid_idx] = clf.predict(valid_x)
        test_pred_proba[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        train_df_pred_proba += clf.predict_proba(train_df[feats])[:, 1] / folds.n_splits

        prediction += \
            clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits

        # df_feature_importance[n_fold] = pd.Series(clf.feature_importances_, index=feats)

        print('Fold %2d AUC : %.6f' % (n_fold, roc_auc_score(valid_y, test_pred_proba[valid_idx])))
        del train_x, train_y, valid_x, valid_y, clf
        gc.collect()

    roc_auc_train = roc_auc_score(train_df['TARGET'], train_pred_proba)
    precision_train = precision_score(train_df['TARGET'], train_pred, average=None)
    recall_train = recall_score(train_df['TARGET'], train_pred, average=None)

    roc_auc_test = roc_auc_score(train_df['TARGET'], test_pred_proba)
    precision_test = precision_score(train_df['TARGET'], test_pred, average=None)
    recall_test = recall_score(train_df['TARGET'], test_pred, average=None)

    print('Full AUC score %.6f' % roc_auc_test)

    df_feature_importance.fillna(0, inplace=True)
    df_feature_importance['mean','sum'] = df_feature_importance.mean(axis=1)

    # Write prediction files
    if save_train_prediction:
        df_prediction = train_df[['SK_ID_CURR', 'TARGET']]
        df_prediction['Prediction'] = test_pred_proba
        df_prediction.to_csv(train_prediction_file_name, index=False)
        del df_prediction
        gc.collect()

    if save_test_prediction:
        df_prediction = test_df[['SK_ID_CURR']]
        df_prediction['TARGET'] = prediction
        df_prediction.to_csv(test_prediction_file_name, index=False)
        del df_prediction
        gc.collect()

    return df_feature_importance, \
           [roc_auc_train, roc_auc_test,
            precision_train[0], precision_test[0], precision_train[1], precision_test[1],
            recall_train[0], recall_test[0], recall_train[1], recall_test[1], 0], train_df_pred_proba


xgb_params = {
    'gpu_id': 0,
    'tree_method': 'gpu_hist',
    'updater': 'grow_gpu',
    'objective': 'binary:logistic',
    'max_depth': 4,
    'predictor': 'cpu_predictor'
}

for max_depth in [4, 6, 8]:
    for n_estimators in [400, 600, 800, 1100]:
        for min_child_weight in [1, 3, 5, 7]:
            xgb_params['max_depth'] = max_depth
            xgb_params['n_estimators'] = n_estimators
            xgb_params['min_child_weight'] = min_child_weight

'''
bureau 的问题：1. days end date fact 是否重复
cc 的问题 amt payment amt payment total minnimum regularity 三者相互联系的，amt payment 是总付钱，（不是很清楚
        credit balance 是负债，credit limit actual 是额度 所以future payment 用错了，有的记录缺了最近一个月的数据
        cc的数据有的最后一个月没有（可能跟账单时间有关）
pos 的问题 貌似没有问题，根据ins table 加入了paymen 因素，pos 的特点是这个loan 用来购买东西，之后分期付款还，credit card
            的特点是申请了credit 以后可以多次取现金（就是credit card 啦）
            
        
'''
