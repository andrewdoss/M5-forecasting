'''
This class provides a means of flexible and convenient model validation.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Validator():
    '''Validator object.

    Parameters
    -----------
    train_df : pandas DataFrame
        The training dataset for the competition.
    val_block : int
        A negative integer index for the 28 day block to use for validation.
    calendar : pandas DataFrame
        The calendar dataset for the competition.
    price : pandas DataFrame
        The price dataset for the competition.
    '''
    def __init__(self, train_df, calendar_df, price_df, val_block=-1):
        self.train = train_df
        self.val_block = val_block
        self.calendar = calendar_df
        self.price = price_df
        self.get_train_val_split(val_block)

    def __str__(self):
        return f"Validator Object: Block={self.val_block}"

    def __repr__(self):
        return f"Validator Object: Block={self.val_block}"

    def get_train_val_split(self, k=-1):
        '''Method for generating training and validation datasets.
        
        Arguments
        ----------
        k : int
            A negative integer index for the 28 date block to hold out.
        
        '''
        self.val_block = k
        assert k < 0, "Specified validation block must be a negative index."
        if k == -1:
            self.train_split = self.train.iloc[:,:28*k].copy()
            self.val_split = pd.concat([self.train['id'], self.train.iloc[:,28*k:].copy()], axis=1)
        else:
            self.train_split = self.train.iloc[:,:28*k].copy()
            self.val_split = pd.concat([self.train['id'], self.train.iloc[:,28*k:28*(k+1)].copy()], axis=1)
        # Recompute weights from training split
        self._get_weights()
        # Reconstruct aggregated training and validation series
        self.train_split_agg = self._agg_time_series(self.train_split)
        self.val_split_agg = self._agg_time_series(self.val_split)

    def _rmsse(self, y_in, y_out, y_hat):
        '''Helper for computing the RMSSE for a single time series'''
        y_in_1d, y_out_1d, y_hat_1d = y_in.squeeze(), y_out.squeeze(), y_hat.squeeze()
        # Truncate y_in_1d to days after first non-zero sales
        y_in_1d = y_in_1d[np.min(np.nonzero(y_in_1d)):]
        num = np.mean((y_out_1d - y_hat_1d)**2)
        den = np.mean(np.diff(y_in_1d)**2)
        # If the insample series is constant, it is likely due to no insample demand which means no weight
        # This return value will be conspicuous in end metric if "no weight" assumption is invalid
        if den == 0:
            score = 1e9
        else:
            #score = np.sqrt(num/(h*den))
            score = np.sqrt(num/den)
        return score

    def _agg_time_series(self, df):
        '''Aggregate the level 12 time series up to a dataframe with all levels
        
        df : pandas DataFrame, must have an "id" attribute for aggregation
        '''
        assert 'id' in df.columns, "Dataframes must have 'id' column for aggregation"
        agg_levels_dict = {12: ['item_id', 'store_id'],
                           11: ['item_id', 'state_id'],
                           10: ['item_id'],
                            9: ['store_id', 'dept_id'],
                            8: ['store_id', 'cat_id'],
                            7: ['state_id', 'dept_id'],
                            6: ['state_id', 'cat_id'],
                            5: ['dept_id'],
                            4: ['cat_id'],
                            3: ['store_id'],
                            2: ['state_id']}
        # Add attributes for aggregation, if needed
        merge_cols = [col for col in list(self.train_split.columns) if 'd_' not in col]
        if {'item_id', 'store_id', 'state_id', 'dept_id', 'cat_id'}.issubset(df.columns):
            mdf = df.copy()
        else:
            mdf = self.train_split.loc[:,merge_cols].merge(df, how='left', on='id')
        agg_df = pd.DataFrame()
        # Perform global aggregation
        grp_df = mdf.drop(columns=merge_cols)
        grp_df = grp_df.sum(axis=0)
        grp_df['id'] = 'global'
        grp_df['level'] = 1
        agg_df = pd.DataFrame(grp_df).T.copy()
        # Perform other levels of aggregation
        for i in range(2,13):
            grp_keys = agg_levels_dict[i]
            grp_df = mdf.loc[:,grp_keys + [col for col in mdf.columns if col not in merge_cols]].groupby(grp_keys, as_index=False).sum()
            grp_df['id'] = ''
            for key in grp_keys:
                grp_df['id'] += grp_df[key] + '_'
            grp_df['id'] = grp_df['id'].str.slice(0,-1)
            grp_df['level'] = i
            grp_df.drop(columns=grp_keys, inplace=True)
            agg_df = agg_df.append(grp_df)
        agg_df.reset_index(drop=True, inplace=True)
        return agg_df

    def _get_weights(self):
        '''Get weight of each series for the WRMSSE'''
        # Get last 28 days of unit sales for the provided training set
        last_train_cols = [col for col in self.train_split.columns if 'd_' not in col] + list(self.train_split.columns[-28:])
        temp_train = self.train.loc[:,last_train_cols].copy()
        # Melt training set to enable joining of prices to unit sales
        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        melt_temp_train = temp_train.melt(id_vars=id_vars, var_name='d', value_name='unit_sales')
        merged_df = melt_temp_train.merge(self.calendar.loc[:,['d', 'wm_yr_wk']], how='left', on='d').merge(self.price, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        merged_df['dollar_sales'] = merged_df['unit_sales'] * merged_df['sell_price']
        # Pivot series back out into wide format and join aggregation features
        pivot_df = merged_df.pivot(index='id', columns='d', values='dollar_sales').reset_index()
        pivot_df = pivot_df.merge(temp_train.loc[:,[col for col in temp_train.columns if 'd_' not in col]], how='left', on='id')
        # Get the level aggregation form of the wide dataframe
        agg_df = self._agg_time_series(pivot_df)
        # Get the dollar sum per series and divide by overall sum
        agg_df['series_sum'] = agg_df.loc[:,[col for col in agg_df.columns if 'd_' in col]].sum(axis=1)
        agg_df['weight'] = agg_df['series_sum'] / np.sum(agg_df['series_sum'])
        self.weights = agg_df.drop(columns=[col for col in agg_df.columns if 'd_' in col])

    def _get_wrmsse(self, pred_df, val_block):
        '''Computes per-series WRMSSE'''
        if (val_block is not None) and (val_block != self.val_block):
            self.get_train_val_split(val_block)
        self.pred_agg = self._agg_time_series(pred_df)
        val_error = []
        all_y_in = self.train_split_agg.drop(columns=['id', 'level']).values
        all_y_out = self.val_split_agg.drop(columns=['id', 'level']).values
        all_yhat = self.pred_agg.drop(columns=['id', 'level']).values
        for i in range(self.val_split_agg.shape[0]):
            val_error.append(self._rmsse(all_y_in[i,:], all_y_out[i,:], all_yhat[i,:]))
        self.pred_agg['rmsse'] = val_error
        self.pred_agg = self.pred_agg.merge(self.weights.loc[:,['id', 'weight']], how='left', on='id')
        self.pred_agg['weighted_rmsse'] = self.pred_agg['rmsse'] * self.pred_agg['weight']
        self.all_wrmsse = self.pred_agg.loc[:,['id', 'level', 'rmsse', 'weight', 'weighted_rmsse']].copy()

    def score_predictions(self, pred_df, val_block=None, verbose=False):
        '''Computes scores per hierarchical level and overall
        
        pred_df : pandas DataFrame
            A pandas DataFrame with predictions that must include an id column.
        val_block : int
            A negative integer index for the 28 day block to use for validation.
        verbose : boolean
            A flag specifying whether the results should be printed as well as returned.
        '''
        def upweighted_sum(x):
            return 12*np.sum(x)
        # Compute scores all all time series
        self._get_wrmsse(pred_df, val_block)
        score = self.all_wrmsse['weighted_rmsse'].sum()
        if verbose:
            self.all_wrmsse.groupby('level').agg({'weighted_rmsse':upweighted_sum}).sort_index().plot(kind='barh')
            plt.show()
            print(f"Weighted RMSSE: {score:.3f}") 
        return self.all_wrmsse['weighted_rmsse'].sum() 
    