# WOE Binning

# Calculation of WOE 
# WOE = ln(%Event/%Non-Event)
# IV = Sum ((%Event - #Non-Event) * WOE) For Each Bucket

# 1. Each bin should have at least 5% of the observations
# 2. Each bin should be non-zero for both good and bad loans
# 3. The WOE should be distinct for each category. Similar groups should be aggregated or binned together. 
#    It is because the bins with similar WoE have almost the same proportion of good or bad loans, implying 
#    the same predictive power
# 4. The WOE should be monotonic, i.e., either growing or decreasing with the bins
# 5. Missing values are binned separately

#%%

import pandas as pd
import sklearn
df_app = pd.read_csv("./home-credit-default-risk/application_train.csv")
df_app.rename(str.lower, axis='columns', inplace = True)
print(f"The shape of the training dataset: {df_app.shape}")

# %% Pick amt_income_total for binning
s_amt_income_total = df_app.amt_income_total.map(lambda x : x/1e5) # make the units into 10's thousands
s_amt_income_total.describe()
# %%
import numpy as np
y = df_app.target
s_amt_income_total_bins = pd.cut(s_amt_income_total, bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, np.inf],
                                labels = [1, 2, 3, 4, 5, 6]) # dividing close to distribution

df = pd.concat({"X": s_amt_income_total, "y": y}, axis = 1)
# %%
df_woe = df.groupby(s_amt_income_total_bins).\
                                    agg(pct_event = ("target", lambda x: sum(x)/len(x)), 
                                        pct_non_event = ("target", lambda x: 1 - sum(x)/len(x)),
                                        n_event = ("target", "sum"),
                                        n_non_event = ("target", lambda x : abs(sum(x-1))),
                                        n_obs = ("target", "count"))

df_woe["woe"] = df_woe.apply(lambda x: np.log(x.pct_event/x.pct_non_event), axis = 1)  
df_woe["pct_event_minus_pct_non_event"] = df_woe.apply(lambda x: x.pct_event - x.pct_non_event, axis = 1)  
df_woe["iv"] = df_woe.apply(lambda x: x.pct_event_minus_pct_non_event * x.woe, axis = 1)
print(df_woe)
print(sum(df_woe.iv))
# %%
# define a binning function
from scipy import stats

# Function to perform binning on a coninues variable. utilisng spearman
# coefficent and p-value to determine suitability. 
def optimal_binning(x, y, bin_range = (5, 10), confidence_threshold = 0.05):
    df = pd.concat({"y": y, "x": x}, axis = 1)
    df_missing = df[x.isna()]
    df_with_values = df[x.notna()] 
    rho = np.nan
    p_value = np.nan
    buckets = np.nan
    df = None
    for n in range(bin_range[0], bin_range[1]):
        print(n)
        df_with_values["bin"] = pd.qcut(x, n, duplicates = "drop")
        #print(df_with_values.head())
        df_bins = df_with_values.groupby("bin")
        r, p = stats.spearmanr(df_bins.mean().x, df_bins.mean().y)
        #print(df_bins.mean().x, df_bins.mean().y)
        print(r, p)
        if (np.isnan(rho) or abs(r) > abs(rho)) and p <= confidence_threshold:
            rho = r
            p_value = p
            buckets = n
            df = df_with_values

    return {"buckets": buckets, "rho": rho, "p_value": p_value, "df": df}

# calculate woe, iv
def calculate_stats(y, x):
    n_obs = len(y)
    n_events = y.sum()
    n_non_events = n_obs - n_events
    woe = np.log(n_events/n_non_events)
    iv = woe * (n_events/n_obs - n_non_events/n_obs)
    n_pct_events = n_events/n_obs

    return pd.Series(np.array([n_obs, n_pct_events, n_events, n_non_events, woe, iv]))

# %%
#ret = optimal_binning(s_amt_income_total, y, (1, 20))
ret = optimal_binning(df_app.amt_credit, y, (1,20))
x = ret["df"]

# %%
x.groupby("bin").apply(lambda x: calculate_stats(x.y, x.x)).sort_values('bin')
# %%
from sklearn.feature_selection import chi2, f_classif, SelectKBest, SelectFromModel
