from pathlib import Path
from eda import EDAApplier
from regression import BatchGradientDescent, KFoldRegression, MiniBatchGradientDescent, RidgeRegression, StohasticGradientDescent

DS_PATH = Path("/Users/sweetferrero/Downloads/insurance.csv")

if __name__ == "__main__":
    eda = EDAApplier(DS_PATH)
    reg = BatchGradientDescent(eda.df)
    reg.descent(epochs=500, alpha=0.01)
    reg.make_prediction()
    print(eda.df)
