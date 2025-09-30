from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cmap import Colormap
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from kan import KAN  # pykan==0.0.5

OUTPUT_DIR = "KAN/kan_classification"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({"font.size": 16})

cmap = "cubehelix"
cm = Colormap(cmap).to_mpl()


def generate_dataset(dataset_type="classification", n_samples=1000):
    if dataset_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=0,
        )
    elif dataset_type == "P_shape":
        # Generate a "P" shaped dataset
        X = []
        y = []

        # Generate points along the vertical lines of "П"
        for _ in range(n_samples // 3):
            x = np.random.uniform(-2, -1)
            y_val = np.random.uniform(-2, 2)
            X.append([x, y_val])
            y.append(1)

            x = np.random.uniform(1, 2)
            y_val = np.random.uniform(-2, 2)
            X.append([x, y_val])
            y.append(1)

        # Generate points along the top horizontal line of "П"
        for _ in range(n_samples // 3):
            x = np.random.uniform(-2, 2)
            y_val = np.random.uniform(1.5, 2)
            X.append([x, y_val])
            y.append(1)

        # Generate random points outside the "П" shape for Class 0
        for _ in range(n_samples):
            x = np.random.uniform(-3, 3)
            y_val = np.random.uniform(-3, 3)
            if not (-2 <= x <= 2 and -2 <= y_val <= 2 and (x < -1 or x > 1 or y_val > 1.5)):
                X.append([x, y_val])
                y.append(0)

        X, y = np.array(X), np.array(y)
    elif dataset_type == "X_shape":
        X, y = [], []
        for _ in range(n_samples // 2):
            x = np.random.uniform(-2, 2)
            y_val = x + np.random.uniform(-0.5, 0.5)
            X.append([x, y_val])
            y.append(0)
        for _ in range(n_samples // 2):
            x = np.random.uniform(-2, 2)
            y_val = -x + np.random.uniform(-0.5, 0.5)
            X.append([x, y_val])
            y.append(1)
        X, y = np.array(X), np.array(y)
    else:
        raise ValueError("Unknown dataset_type")
    return train_test_split(X, y, test_size=0.2, random_state=0)


def convert_to_kan_dataset(X_train, X_val, y_train, y_val):
    X_train_tensor = torch.tensor(X_train, dtype=torch.get_default_dtype())
    X_val_tensor = torch.tensor(X_val, dtype=torch.get_default_dtype())
    y_train_tensor = torch.tensor(y_train, dtype=torch.get_default_dtype()).reshape(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.get_default_dtype()).reshape(-1, 1)
    return {
        "train_input": X_train_tensor,
        "train_label": y_train_tensor,
        "test_input": X_val_tensor,
        "test_label": y_val_tensor,
    }


class KANBoostingClassifier:
    """KANBoostingClassifier is a boosting classifier based on KAN."""

    def __init__(
        self,
        n_estimators=10,
        n_steps=10,
        learning_rate=0.1,
        width=None,
        grid=3,
        k=3,
        lamb=5e-5,
        lamb_entropy=2.0,
        beta=10,
        img_folder=OUTPUT_DIR,
    ):
        if width is None:
            width = [2, 3, 2, 1]
        self.n_estimators = n_estimators
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.width = width
        self.grid = grid
        self.k = k
        self.lamb = lamb
        self.lamb_entropy = lamb_entropy
        self.beta = beta
        self.img_folder = img_folder
        self.models = []
        Path(img_folder).mkdir(parents=True, exist_ok=True)

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot_steps=False):
        self.X_train_data = X_train
        self.y_train_data = y_train
        self.X_val_data = X_val if X_val is not None else X_train
        self.y_val_data = y_val if y_val is not None else y_train
        dataset = convert_to_kan_dataset(self.X_train_data, self.X_val_data, self.y_train_data, self.y_val_data)

        X_train_tensor = dataset["train_input"]
        y_train_tensor = dataset["train_label"]

        model = KAN(width=self.width, grid=self.grid, k=self.k, seed=0)
        model.train(
            dataset,
            opt="LBFGS",
            steps=self.n_steps,
            lamb=self.lamb,
            lamb_entropy=self.lamb_entropy,
            save_fig=False,
            beta=self.beta,
            in_vars=[r"$x_1$", r"$x_2$"],
            out_vars=[r"Class"],
            img_folder=f"{self.img_folder}_0",
        )
        self.models.append(model)

        for i in range(1, self.n_estimators):
            with torch.no_grad():
                probs = torch.sigmoid(self.predict_logits(X_train_tensor))
                residuals = y_train_tensor - probs

            residual_dataset = {
                "train_input": X_train_tensor,
                "train_label": residuals,
                "test_input": dataset["test_input"],
                "test_label": dataset["test_label"] - torch.sigmoid(self.predict_logits(dataset["test_input"])),
            }

            model = KAN(width=self.width, grid=self.grid, k=self.k, seed=i)
            model.train(
                residual_dataset,
                opt="LBFGS",
                steps=self.n_steps,
                lamb=self.lamb,
                lamb_entropy=self.lamb_entropy,
                save_fig=False,
                beta=self.beta,
                in_vars=[r"$x_1$", r"$x_2$"],
                out_vars=[f"Residual {i}"],
                img_folder=f"{self.img_folder}_{i}",
            )
            self.models.append(model)

    def predict_logits(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        logits = self.models[0].forward(X)
        for i in range(1, len(self.models)):
            logits += self.models[i].forward(X) * self.learning_rate
        return logits

    def predict_proba(self, X):
        logits = self.predict_logits(X)
        return torch.sigmoid(logits).detach().numpy()

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def generate_animation(self, save_path=None, fps=2, dpi=150):
        animator = AnimatedKANBoosting(trained_model=self, img_folder=self.img_folder)
        return animator.create_animation(fps=fps, dpi=dpi, save_path=save_path)


class AnimatedKANBoosting:
    """Class to create an animation of the KANBoosting decision boundary."""

    def __init__(self, trained_model, img_folder=OUTPUT_DIR):
        self.model = trained_model
        self.X_train = trained_model.X_train_data
        self.y_train = trained_model.y_train_data
        self.n_estimators = len(trained_model.models)
        self.learning_rate = trained_model.learning_rate
        self.img_folder = img_folder

    def create_animation(self, fps=2, dpi=100, save_path=None):
        """Create an animation of the KANBoosting decision boundary."""
        if save_path is None:
            save_path = f"{self.img_folder}/animation.gif"

        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        h = 0.02
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        def update(frame):
            ax.clear()
            active_models = self.model.models[: frame + 1]

            def predict_proba(X):
                X_tensor = torch.tensor(X, dtype=torch.float32)
                logits = active_models[0].forward(X_tensor)
                for i in range(1, len(active_models)):
                    logits += active_models[i].forward(X_tensor) * self.learning_rate
                probs = torch.sigmoid(logits)
                return probs.detach().numpy()

            Z = predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
            ax.contour(xx, yy, Z, colors="black", levels=[0.5], linestyles="dashed")
            ax.scatter(
                self.X_train[self.y_train == 0, 0],
                self.X_train[self.y_train == 0, 1],
                color="#26828e",
                edgecolor="k",
                s=40,
            )
            ax.scatter(
                self.X_train[self.y_train == 1, 0],
                self.X_train[self.y_train == 1, 1],
                color="#db2bb6",
                edgecolor="k",
                s=40,
            )
            ax.set_title(f"KANBoosting Decision Boundary (Iteration {frame + 1}/{self.n_estimators})")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

        anim = FuncAnimation(fig, update, frames=list(range(self.n_estimators)), interval=1000 // fps, blit=False)
        anim.save(save_path, writer="imagemagick")
        plt.close()
        return anim


if __name__ == "__main__":
    DATASET_TYPE = "P_shape"  # Change to "classification" for classification dataset
    N_ESTIMATORS = 10
    N_SAMPLES = 1000
    N_STEPS = 10  # for faster training
    FPS = 1

    X_train, X_val, y_train, y_val = generate_dataset(dataset_type=DATASET_TYPE, n_samples=N_SAMPLES)

    kan_clf = KANBoostingClassifier(n_estimators=N_ESTIMATORS, n_steps=N_STEPS)
    kan_clf.fit(X_train, y_train, X_val, y_val, plot_steps=True)
    kan_clf.generate_animation(save_path=f"{OUTPUT_DIR}/kan_boosting_animation_{DATASET_TYPE}.gif", fps=FPS)

    y_pred = kan_clf.predict(X_val)
    print(f"Final Test Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Final Test AUC: {roc_auc_score(y_val, kan_clf.predict_proba(X_val)):.4f}")
