from abc import abstractmethod
from functools import cached_property
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score



class BaseLinearRegression:

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.l2_lambda: float = 0.001
        self.weights = np.zeros(len(self.dataset.columns) - 1).reshape(-1, 1)

        self.y_true = dataset["charges"].to_numpy().reshape(-1, 1)
        X_ = dataset.drop(columns=["charges"])

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        X_scaled = self.x_scaler.fit_transform(X_)
        y_scaled = self.y_scaler.fit_transform(self.y_true)
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X_scaled,
            y_scaled,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )
        self.bias = 0

    def loss_function(self, y_pred, y_true, weights):
        mse_loss = np.mean((y_pred - y_true) ** 2)
        l2_penalty = self.l2_lambda * np.sum(weights**2)
        return mse_loss + l2_penalty

    def loss_without_l2(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @cached_property
    def lsm_weights(self):
        return (
            np.linalg.inv(
                self._X_train.T @ self._X_train
                + self.l2_lambda * np.eye(self._X_train.shape[1])
            )
            @ self._X_train.T
            @ self._y_train
        )

    def least_squares_predict(self):
        y_pred = self._X_test @ self.lsm_weights
        loss = self.loss_without_l2(y_pred, self._y_test)
        inversed = self.y_scaler.inverse_transform(y_pred)
        print("Loss (Least Squares): ", loss)
        return inversed

    @abstractmethod
    def descent(self, epochs=500, alpha=0.01):
        """
        Abstract method to implement the descent algorithm for the given linear regression model.

        Parameters:
        epochs (int): The number of epochs to train the model. Defaults to 500.
        alpha (float): The learning rate for the gradient descent. Defaults to 0.01.
        """
        ...

    def make_prediction(self):
        y_pred = self._X_test @ self.weights + self.bias
        loss = self.loss_without_l2(y_pred, self._y_test)
        inversed = self.y_scaler.inverse_transform(y_pred)
        print("Loss (MSE): ", loss)
        print("RMSE: ", rmse(self._y_test, y_pred))
        print("R2: ", r2_score(self._y_test, y_pred))
        return inversed


class StohasticGradientDescent(BaseLinearRegression):

    def _stohastic_loss_weights_gradient(self, y_true, y_pred, x):
        return x.reshape(-1, 1) * (y_pred - y_true)

    def _stohastic_loss_bias_gradient(
        self,
        y_true,
        y_pred,
    ):
        return y_pred - y_true

    def descent(self, epochs=100, alpha=0.01):
        for epoch in range(1, epochs):
            indices = np.arange(len(self._X_train))
            np.random.shuffle(indices)

            self._X_train = self._X_train[indices]
            self._y_train = self._y_train[indices]

            epoch_loss = 0
            for row, y_actual in zip(self._X_train, self._y_train):
                y_i_pred = np.array(row) @ self.weights + self.bias
                dw = self._stohastic_loss_weights_gradient(y_actual, y_i_pred, row)
                db = self._stohastic_loss_bias_gradient(y_actual, y_i_pred)
                self.weights -= alpha * dw
                self.bias -= alpha * db
                epoch_loss += self.loss_without_l2(y_i_pred, y_actual)


class BatchGradientDescent(BaseLinearRegression):

    def loss_derivative_over_weights(self, y_true, y_pred, x):
        return (
            2 / len(y_true) * (x.T @ (y_pred - y_true))
            + 2 * self.l2_lambda * self.weights
        )

    def loss_derivative_over_bias(
        self,
        y_true,
        y_pred,
    ):
        return 2 / len(y_true) * np.sum(y_pred - y_true)

    def descent(self, epochs=500, alpha=0.01):
        for epoch in range(1, epochs):
            y_pred = self._X_train @ self.weights
            loss = self.loss_function(y_pred, self._y_train, self.weights)
            print(f"Epoch: {epoch}, Loss: {loss}")
            dw = self.loss_derivative_over_weights(self._y_train, y_pred, self._X_train)
            db = self.loss_derivative_over_bias(self._y_train, y_pred)
            self.weights -= alpha * dw
            self.bias -= alpha * db


class MiniBatchGradientDescent(BatchGradientDescent):

    def __init__(self, dataset, batch_size: int = 10, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.batch_size = batch_size

    def create_batches(self, batch_size):
        n_samples = len(self._X_train)

        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = self._X_train[indices]
        y = self._y_train[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            yield X_batch, y_batch

    def descent(self, epochs=500, alpha=0.01):
        for epoch in range(1, epochs):
            epoch_loss = 0
            for X_batch, y_batch in self.create_batches(self.batch_size):
                y_pred = X_batch @ self.weights + self.bias
                loss = self.loss_function(y_pred, y_batch, self.weights)
                print(f"Epoch: {epoch}, Loss: {loss}")
                dw = self.loss_derivative_over_weights(y_batch, y_pred, X_batch)
                db = self.loss_derivative_over_bias(y_batch, y_pred)
                self.weights -= alpha * dw
                self.bias -= alpha * db

            epoch_loss /= len(self._X_train) / self.batch_size
            print(f"Epoch {epoch}: avg loss {epoch_loss}")


class RidgeRegression(BaseLinearRegression):

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.ridge = Ridge()

    def descent(self, *args, **kwargs):
        self.ridge.fit(self._X_train, self._y_train)
        self.weights = self.ridge.coef_
        self.bias = self.ridge.intercept_

    def make_prediction(self):
        y_train_pred = self.ridge.predict(self._X_train)
        y_test_pred = self.ridge.predict(self._X_test)

        mse_train = mse(self._y_train, y_train_pred)
        mse_test = mse(self._y_test, y_test_pred)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        r2_train = r2_score(self._y_train, y_train_pred)
        r2_test = r2_score(self._y_test, y_test_pred)

        print(f"MSE train: {mse_train}, MSE test: {mse_test}")
        print(f"RMSE train: {rmse_train}, RMSE test: {rmse_test}")
        print(f"R2 train: {r2_train}, R2 test: {r2_test}")


class KFoldRegression(BaseLinearRegression):

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.metrics = {
            "mse_train": [],
            "mse_test": [],
            "rmse_train": [],
            "rmse_test": [],
            "r2_train": [],
            "r2_test": [],
        }

    def descent(self, *args, **kwargs):
        for train_index, test_index in self.kf.split(self._X_train):
            X_tr, X_val = self._X_train[train_index], self._X_train[test_index]
            y_tr, y_val = self._y_train[train_index], self._y_train[test_index]

            model = Ridge(alpha=0.01)
            model.fit(X_tr, y_tr)

            y_tr_pred = model.predict(X_tr)
            y_val_pred = model.predict(X_val)

            self.metrics["mse_train"].append(mse(y_tr, y_tr_pred))
            self.metrics["mse_test"].append(mse(y_val, y_val_pred))
            self.metrics["rmse_train"].append(np.sqrt(self.metrics["mse_train"][-1]))
            self.metrics["rmse_test"].append(np.sqrt(self.metrics["mse_test"][-1]))
            self.metrics["r2_train"].append(r2_score(y_tr, y_tr_pred))
            self.metrics["r2_test"].append(r2_score(y_val, y_val_pred))

    def make_table(self):
        k = len(self.metrics["mse_train"])

        # Собираем DataFrame
        table = pd.DataFrame(
            {
                f"Fold{i+1}": [
                    self.metrics["mse_train"][i],
                    self.metrics["mse_test"][i],
                    self.metrics["rmse_train"][i],
                    self.metrics["rmse_test"][i],
                    self.metrics["r2_train"][i],
                    self.metrics["r2_test"][i],
                ]
                for i in range(k)
            },
            index=[
                "mse-train",
                "mse-test",
                "rmse-train",
                "rmse-test",
                "r2-train",
                "r2-test",
            ],
        )

        # Добавляем столбцы E (mean) и STD
        table["Mean"] = table.mean(axis=1)
        table["STD"] = table.std(axis=1)

        # Транспонируем, чтобы строки были фолдами (по желанию)
        table = table.T
        print(table)


def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))
