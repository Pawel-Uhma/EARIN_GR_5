import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from datasets import load_dataset

DATASET = "wine"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_and_preprocess_data(seed):
    X, y = load_dataset(DATASET)
    if hasattr(X, "columns"):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names

def plot_pre_training_data(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    for class_value in np.unique(y):
        plt.scatter(X_pca[y == class_value, 0], X_pca[y == class_value, 1],
                    label=f'Class {class_value}', alpha=0.7)
    plt.title("PCA Projection of Wine Data (Pre-training)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/pre_training_pca.png")
    plt.close()

def plot_feature_importances(importances, feature_names, title, filename):
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
    plt.xticks(range(len(sorted_importances)), sorted_features, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"./plots/{filename}")
    plt.close()

def plot_confusion_matrix(cm, class_names, title, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"./plots/{filename}")
    plt.close()

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    class_names = [f'Class {i}' for i in np.unique(y_test)]
    plot_confusion_matrix(cm, class_names, f"Confusion Matrix: {model.__class__.__name__}",
                          f"{model.__class__.__name__}_confusion.png")
    return acc, predictions

def plot_test_pca_with_predictions(X_test, y_test, predictions, best_model_name):
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)
    plt.figure(figsize=(8, 6))
    for class_value in np.unique(y_test):
        plt.scatter(X_test_pca[y_test == class_value, 0], X_test_pca[y_test == class_value, 1],
                    label=f'True Class {class_value}', alpha=0.6, marker='o')
    for class_value in np.unique(predictions):
        plt.scatter(X_test_pca[predictions == class_value, 0], X_test_pca[predictions == class_value, 1],
                    label=f'Predicted Class {class_value}', alpha=0.3, marker='x')
    plt.title(f"PCA Projection with Predictions ({best_model_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./plots/{best_model_name}_test_pca.png")
    plt.close()

def main():
    set_seed(0)
    os.makedirs("./plots", exist_ok=True)
    X, y, feature_names = load_and_preprocess_data(0)
    plot_pre_training_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model_dt = DecisionTreeClassifier(max_depth=4, random_state=0)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=0)
    scores_dt = cross_val_score(model_dt, X_train, y_train, cv=4, scoring='accuracy')
    scores_rf = cross_val_score(model_rf, X_train, y_train, cv=4, scoring='accuracy')
    final_model_dt = model_dt.fit(X_train, y_train)
    final_model_rf = model_rf.fit(X_train, y_train)
    acc_dt, _ = evaluate_model(final_model_dt, X_test, y_test)
    acc_rf, _ = evaluate_model(final_model_rf, X_test, y_test)
    if hasattr(final_model_dt, "feature_importances_"):
        plot_feature_importances(final_model_dt.feature_importances_, feature_names,
                                 "Decision Tree Feature Importances",
                                 "DecisionTree_feature_importances.png")
    if hasattr(final_model_rf, "feature_importances_"):
        plot_feature_importances(final_model_rf.feature_importances_, feature_names,
                                 "Random Forest Feature Importances",
                                 "RandomForest_feature_importances.png")
    best_model = final_model_dt if acc_dt >= acc_rf else final_model_rf
    best_model_name = best_model.__class__.__name__
    _, best_predictions = evaluate_model(best_model, X_test, y_test)
    plot_test_pca_with_predictions(X_test, y_test, best_predictions, best_model_name)

if __name__ == "__main__":
    main()
