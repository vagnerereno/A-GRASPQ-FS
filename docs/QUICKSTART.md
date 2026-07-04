# Quickstart

```python
from agraspqfs import AGraspQFeatureSelector
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

selector = AGraspQFeatureSelector(
    estimator=GaussianNB(),
    preset="fast",
    min_features=2,
    max_features=15,
    random_state=42,
)

selector.fit(X_train, y_train)
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

clf = GaussianNB().fit(X_train_sel, y_train)
y_pred = clf.predict(X_test_sel)

print(selector.selected_features_)
print(selector.n_features_selected_)
print(f1_score(y_test, y_pred))
```

## CLI example

```bash
agraspqfs   --csv data/hibrid_dataset_GOOSE_train.csv   --target class   --estimator nb   --preset fast   --min-features 2   --max-features 15   --output results/agraspqfs_selection.json   --history-csv results/agraspqfs_history.csv
```
