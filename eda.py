import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from pathlib import Path


class EDAApplier:

    def __init__(self, path: Path):
        self.df: pd.DataFrame = pd.read_csv(path, index_col=False)
        pairs = [("sex", "male"), ("smoker", "yes"), ("region", "southwest")]
        for field, one in pairs:
            self._encode_binary(field=field, one=one)
        #self._show_correlation(self.df)

    def _show_correlation(self, df: pd.DataFrame):
        _, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), cmap="Reds", ax=ax)
        plt.show()

    def _encode_binary(self, *, field, one):
        self.df[field] = self.df.apply(
            lambda row: 1 if row[field] == one else 0, axis=1
        )

    def _fill_na_with_average(self, *fields, method="mean"):
        for field in set(fields):
            nan_free = self.df[field].dropna()
            method = getattr(nan_free, method)
            self.df[field] = self.df[field].fillna(method())
