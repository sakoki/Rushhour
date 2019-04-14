from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize(df, with_std=False):
    # fill na
    df = df.fillna(0)
    # scale data. output: ndarray
    scaler = StandardScaler(with_std=with_std)

    score_scale = scaler.fit_transform(df)
    # turn scaled data into df.
    df_scale = pd.DataFrame(score_scale, index=df.index, columns=df.columns)

    return df_scale