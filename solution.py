import os
import json

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


def _stratify(df):
    df['strat_age'] = 5
    df.loc[(df['age'] > 23) & (df['age'] <= 25), 'strat_age'] = 1
    df.loc[(df['age'] > 25) & (df['age'] <= 27), 'strat_age'] = 2
    df.loc[(df['age'] > 27) & (df['age'] <= 30), 'strat_age'] = 3
    df.loc[df['age'] > 30, 'strat_age'] = 4

    df['strat'] = df.groupby(['strat_age', 'gender']).ngroup()
    w = (df['strat'].value_counts() / df.shape[0]).to_dict()

    return df.drop('strat_age', axis=1), w


df_users = pd.read_csv(os.environ['PATH_DF_USERS'])
df_sales = pd.read_csv(os.environ['PATH_DF_SALES'])

df_users, weights = _stratify(df_users)

df_sales = df_sales.loc[df_sales['day'].isin(np.arange(42, 55))]
df_sales = df_sales.groupby(['day', 'user_id'], as_index=False)['sales'].sum().astype(np.int)

df_prepilot = (df_sales
               .loc[df_sales['day'].isin(np.arange(42, 48))]
               .groupby('user_id', as_index=False)
               .agg(sales_sum_prepilot=('sales', 'sum'), sales_cnt_prepilot=('sales', 'count')))
df_prepilot[['sales_sum_prepilot', 'sales_cnt_prepilot']] = df_prepilot[['sales_sum_prepilot', 'sales_cnt_prepilot']].astype(np.int).clip(upper=3704)

df_pilot = (df_sales
            .loc[df_sales['day'].isin(np.arange(49, 55))]
            .groupby('user_id', as_index=False)
            .agg(sales_sum=('sales', 'sum'), sales_cnt=('sales', 'count')))
df_pilot[['sales_sum', 'sales_cnt']] = df_pilot[['sales_sum', 'sales_cnt']].astype(np.int).clip(upper=3704)

df_pilot = df_users.merge(df_prepilot, how='left', on='user_id').merge(df_pilot, how='left', on='user_id').fillna(0)


app = Flask(__name__)


@app.route('/ping')
def ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']
    has_effect = _check_test(test)
    return jsonify(has_effect=int(has_effect))


def _check_test(test):
    group_a_one, group_a_two, group_b = test['group_a_one'], test['group_a_two'], test['group_b']
    user_a = group_a_one + group_a_two
    user_b = group_b

    df = df_pilot.loc[df_pilot['user_id'].isin(user_a + user_b)]

    df = _linearized_metric(df, users=user_a)
    df = _cuped_metric(df, metric_name='metric_lin', covar_name='covar_lin')

    df = df.drop(
        ['sales_sum_prepilot', 'sales_cnt_prepilot', 'sales_sum', 'sales_cnt', 'metric_lin', 'covar_lin', 'age', 'gender'],
        axis=1
    )

    return _run_post_strat_test(df, user_a, user_b, 'strat', 'metric_lin_cuped', weights)


def _run_post_strat_test(df, user_a, user_b, strat_column, target_name, w):
    mean_strat_pilot = _strat_mean(df[df['user_id'].isin(user_b)], strat_column, target_name, w)
    mean_strat_control = _strat_mean(df[df['user_id'].isin(user_a)], strat_column, target_name, w)
    var_strat_pilot = _strat_var(df[df['user_id'].isin(user_b)], strat_column, target_name, w)
    var_strat_control = _strat_var(df[df['user_id'].isin(user_a)], strat_column, target_name, w)

    delta_mean_strat = mean_strat_pilot - mean_strat_control
    std_mean_strat = (var_strat_pilot / len(user_b) + var_strat_control / len(user_a)) ** 0.5

    confidence_interval = (delta_mean_strat - 1.96 * std_mean_strat, delta_mean_strat + 1.96 * std_mean_strat)

    if min(confidence_interval) > 0 or max(confidence_interval) < 0:
        return 1
    return 0


def _linearized_metric(df, users):
    mask = df['user_id'].isin(users)
    kappa = df.loc[mask, 'sales_sum'].values.sum() / df.loc[mask, 'sales_cnt'].values.sum()

    df['metric_lin'] = (df['sales_sum'] - kappa * df['sales_cnt']).fillna(0)
    df['covar_lin'] = (df['sales_sum_prepilot'] - kappa * df['sales_cnt_prepilot']).fillna(0)
    return df


def _cuped(y, y_cov):
    theta = np.cov(y, y_cov)[0, 1] / np.var(y_cov)
    y_cup = y - theta * y_cov
    return y_cup


def _cuped_metric(df, metric_name, covar_name):
    df[f'{metric_name}_cuped'] = _cuped(df[metric_name].values, df[covar_name].values)
    return df


def _strat_mean(df: pd.DataFrame, strat_column: str, target_name: str, w: dict):
    strat_mean = df.groupby(strat_column)[target_name].mean()
    return (strat_mean * pd.Series(w)).sum()


def _strat_var(df: pd.DataFrame, strat_column: str, target_name: str, w: dict):
    strat_var = df.groupby(strat_column)[target_name].var()
    return (strat_var * pd.Series(w)).sum()
