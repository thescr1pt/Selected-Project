import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss


# Step 1: Preprocessing

X_train = pd.read_csv('dataset/train/X_train.txt', sep=r'\s+', header=None)
X_test = pd.read_csv('dataset/test/X_test.txt', sep=r'\s+', header=None)
y_train = pd.read_csv('dataset/train/y_train.txt', sep=r'\s+', header=None)
y_test = pd.read_csv('dataset/test/y_test.txt', sep=r'\s+', header=None)

activity_labels = pd.read_csv('dataset/activity_labels.txt', sep=' ', header=None, names=['id', 'activity'])
print("Activity Labels:")
print(activity_labels)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(f"\nOriginal X_train shape: {X_train.shape}")
print(f"Original X_test shape: {X_test.shape}")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
print(f"\nEncoded labels: {label_encoder.classes_}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nAfter PCA - X_train shape: {X_train_pca.shape}")
print(f"After PCA - X_test shape: {X_test_pca.shape}")
print(f"Number of components: {pca.n_components_}")
print(f"Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}")

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_pca, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)



# Step 2: Model Training and Evaluation

# Model 1: Logistic Regression (SGDClassifier)

print("\n" + "="*50)
print("Logistic Regression Model (SGDClassifier)")
print("="*50)

learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
n_epochs = 50

best_lr = 0.0001
best_loss = float('inf')
best_train_loss_history = []
best_val_loss_history = []

for lr in learning_rates:
    clf = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=lr, random_state=42)
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(n_epochs):
        clf.partial_fit(X_train_split, y_train_split, classes=np.unique(y_train_encoded))
        
        y_train_prob = clf.predict_proba(X_train_split)
        train_loss = log_loss(y_train_split, y_train_prob)
        train_loss_history.append(train_loss)
        
        y_val_prob = clf.predict_proba(X_val_split)
        val_loss = log_loss(y_val_split, y_val_prob)
        val_loss_history.append(val_loss)
    
    print(f"Learning rate {lr:.4f} → Final Val Log Loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_lr = lr
        best_train_loss_history = train_loss_history
        best_val_loss_history = val_loss_history

print(f"\nBest learning rate: {best_lr} with Validation Log Loss: {best_loss:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), best_train_loss_history, label="Training Log Loss")
plt.plot(range(1, n_epochs + 1), best_val_loss_history, label="Validation Log Loss")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title(f"Logistic Regression - Complexity Curve (Best lr={best_lr})")
plt.legend()
plt.grid(True)
plt.savefig('graphs/Logistic_Regression_Complexity_Curve.png')
plt.show()

lr_final = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=best_lr, random_state=42)
lr_final.fit(X_train_pca, y_train_encoded)

y_test_pred_lr = lr_final.predict(X_test_pca)
y_test_pred_proba_lr = lr_final.predict_proba(X_test_pca)

lr_accuracy = accuracy_score(y_test_encoded, y_test_pred_lr)
lr_loss = log_loss(y_test_encoded, y_test_pred_proba_lr)
lr_precision = precision_score(y_test_encoded, y_test_pred_lr, average='weighted')
lr_mse = np.mean((y_test_encoded - y_test_pred_lr) ** 2)

print(f"\nLogistic Regression Results:")
print(f"Accuracy: {lr_accuracy:.4f}")
print(f"Loss: {lr_loss:.4f}")
print(f"Precision (weighted): {lr_precision:.4f}")
print(f"MSE: {lr_mse:.4f}")


# Model 2: Random Forest (with Complexity Curve)

print("\n" + "="*50)
print("Random Forest Model")
print("="*50)

depths = range(1, 21)
train_loss_history = []
val_loss_history = []

for depth in depths:
    rf_model = RandomForestClassifier(max_depth=depth, random_state=42)
    rf_model.fit(X_train_split, y_train_split)
    
    y_train_pred_proba = rf_model.predict_proba(X_train_split)
    train_loss = log_loss(y_train_split, y_train_pred_proba)
    train_loss_history.append(train_loss)
    
    y_val_pred_proba = rf_model.predict_proba(X_val_split)
    val_loss = log_loss(y_val_split, y_val_pred_proba)
    val_loss_history.append(val_loss)

best_depth = depths[np.argmin(val_loss_history)]
print(f"Best depth: {best_depth}")

rf_final = RandomForestClassifier(max_depth=best_depth, random_state=42)
rf_final.fit(X_train_pca, y_train_encoded)

y_test_pred_rf = rf_final.predict(X_test_pca)
y_test_pred_proba_rf = rf_final.predict_proba(X_test_pca)

rf_accuracy = accuracy_score(y_test_encoded, y_test_pred_rf)
rf_loss = log_loss(y_test_encoded, y_test_pred_proba_rf)
rf_precision = precision_score(y_test_encoded, y_test_pred_rf, average='weighted')

print(f"\nRandom Forest Results:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Loss: {rf_loss:.4f}")
print(f"Precision (weighted): {rf_precision:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(depths, train_loss_history, marker='o', label='Train Loss')
plt.plot(depths, val_loss_history, marker='s', label='Validation Loss')
plt.xlabel('Tree Depth')
plt.ylabel('Log Loss')
plt.title('Random Forest - Complexity Curve (Training vs Validation Loss)')
plt.legend()
plt.savefig('graphs/Random_Forest_Complexity_Curve.png')
plt.show()

print("\n" + "="*50)
print("Model Comparison Summary")
print("="*50)
print(f"\n{'Model':<25} {'Accuracy':<12} {'Loss':<12} {'Precision':<12}")
print("-" * 60)
print(f"{'Logistic Regression':<25} {lr_accuracy:<12.4f} {lr_loss:<12.4f} {lr_precision:<12.4f}")
print(f"{'Random Forest':<25} {rf_accuracy:<12.4f} {rf_loss:<12.4f} {rf_precision:<12.4f}")
