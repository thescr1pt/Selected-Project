import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


X_train_feat = pd.read_csv('dataset/train/X_train.txt', sep=r'\s+', header=None)
X_test_feat = pd.read_csv('dataset/test/X_test.txt', sep=r'\s+', header=None)
y_train_feat = pd.read_csv('dataset/train/y_train.txt', sep=r'\s+', header=None)
y_test_feat = pd.read_csv('dataset/test/y_test.txt', sep=r'\s+', header=None)

X_train_feat = X_train_feat.values
X_test_feat = X_test_feat.values
y_train_feat = y_train_feat.values.ravel()
y_test_feat = y_test_feat.values.ravel()

print(f"\nPre-extracted features X_train shape: {X_train_feat.shape}")
print(f"Pre-extracted features X_test shape: {X_test_feat.shape}")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_feat)
y_test_encoded = label_encoder.transform(y_test_feat)

scaler_feat = StandardScaler()
X_train_scaled = scaler_feat.fit_transform(X_train_feat)
X_test_scaled = scaler_feat.transform(X_test_feat)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"After PCA - X_train shape: {X_train_pca.shape}")
print(f"Number of components: {pca.n_components_}")

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_pca, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)


body_acc_x_train = pd.read_csv('dataset/train/Inertial Signals/body_acc_x_train.txt', sep=r'\s+', header=None)
body_acc_y_train = pd.read_csv('dataset/train/Inertial Signals/body_acc_y_train.txt', sep=r'\s+', header=None)
body_acc_z_train = pd.read_csv('dataset/train/Inertial Signals/body_acc_z_train.txt', sep=r'\s+', header=None)
body_gyro_x_train = pd.read_csv('dataset/train/Inertial Signals/body_gyro_x_train.txt', sep=r'\s+', header=None)
body_gyro_y_train = pd.read_csv('dataset/train/Inertial Signals/body_gyro_y_train.txt', sep=r'\s+', header=None)
body_gyro_z_train = pd.read_csv('dataset/train/Inertial Signals/body_gyro_z_train.txt', sep=r'\s+', header=None)
total_acc_x_train = pd.read_csv('dataset/train/Inertial Signals/total_acc_x_train.txt', sep=r'\s+', header=None)
total_acc_y_train = pd.read_csv('dataset/train/Inertial Signals/total_acc_y_train.txt', sep=r'\s+', header=None)
total_acc_z_train = pd.read_csv('dataset/train/Inertial Signals/total_acc_z_train.txt', sep=r'\s+', header=None)

body_acc_x_test = pd.read_csv('dataset/test/Inertial Signals/body_acc_x_test.txt', sep=r'\s+', header=None)
body_acc_y_test = pd.read_csv('dataset/test/Inertial Signals/body_acc_y_test.txt', sep=r'\s+', header=None)
body_acc_z_test = pd.read_csv('dataset/test/Inertial Signals/body_acc_z_test.txt', sep=r'\s+', header=None)
body_gyro_x_test = pd.read_csv('dataset/test/Inertial Signals/body_gyro_x_test.txt', sep=r'\s+', header=None)
body_gyro_y_test = pd.read_csv('dataset/test/Inertial Signals/body_gyro_y_test.txt', sep=r'\s+', header=None)
body_gyro_z_test = pd.read_csv('dataset/test/Inertial Signals/body_gyro_z_test.txt', sep=r'\s+', header=None)
total_acc_x_test = pd.read_csv('dataset/test/Inertial Signals/total_acc_x_test.txt', sep=r'\s+', header=None)
total_acc_y_test = pd.read_csv('dataset/test/Inertial Signals/total_acc_y_test.txt', sep=r'\s+', header=None)
total_acc_z_test = pd.read_csv('dataset/test/Inertial Signals/total_acc_z_test.txt', sep=r'\s+', header=None)

y_train_raw = pd.read_csv('dataset/train/y_train.txt', sep=r'\s+', header=None)
y_test_raw = pd.read_csv('dataset/test/y_test.txt', sep=r'\s+', header=None)

X_train_raw = np.stack([
    body_acc_x_train.values, body_acc_y_train.values, body_acc_z_train.values,
    body_gyro_x_train.values, body_gyro_y_train.values, body_gyro_z_train.values,
    total_acc_x_train.values, total_acc_y_train.values, total_acc_z_train.values
], axis=2)

X_test_raw = np.stack([
    body_acc_x_test.values, body_acc_y_test.values, body_acc_z_test.values,
    body_gyro_x_test.values, body_gyro_y_test.values, body_gyro_z_test.values,
    total_acc_x_test.values, total_acc_y_test.values, total_acc_z_test.values
], axis=2)

y_train_raw = y_train_raw.values.ravel() - 1
y_test_raw = y_test_raw.values.ravel() - 1

print(f"Raw data X_train shape: {X_train_raw.shape}")
print(f"Raw data X_test shape: {X_test_raw.shape}")

scaler_raw = StandardScaler()
X_train_raw_reshaped = X_train_raw.reshape(-1, X_train_raw.shape[2])
X_test_raw_reshaped = X_test_raw.reshape(-1, X_test_raw.shape[2])
X_train_raw_scaled = scaler_raw.fit_transform(X_train_raw_reshaped)
X_test_raw_scaled = scaler_raw.transform(X_test_raw_reshaped)
X_train_raw = X_train_raw_scaled.reshape(X_train_raw.shape)
X_test_raw = X_test_raw_scaled.reshape(X_test_raw.shape)

X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
    X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw
)

# =============================================================================
# Logistic Regression 
# =============================================================================

print("\n" + "="*60)
print("Training Model 1: Logistic Regression (lr=0.001)")
print("="*60)

lr = 0.001
n_epochs = 50

lr_clf = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=lr, random_state=42)

lr_train_loss_history = []
lr_val_loss_history = []

for epoch in range(n_epochs):
    lr_clf.partial_fit(X_train_split, y_train_split, classes=np.unique(y_train_encoded))
    
    y_train_prob = lr_clf.predict_proba(X_train_split)
    train_loss = log_loss(y_train_split, y_train_prob)
    lr_train_loss_history.append(train_loss)
    
    y_val_prob = lr_clf.predict_proba(X_val_split)
    val_loss = log_loss(y_val_split, y_val_prob)
    lr_val_loss_history.append(val_loss)

lr_final = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=lr, random_state=42)
lr_final.fit(X_train_pca, y_train_encoded)

y_test_pred_lr = lr_final.predict(X_test_pca)
y_test_pred_proba_lr = lr_final.predict_proba(X_test_pca)

lr_accuracy = accuracy_score(y_test_encoded, y_test_pred_lr)
lr_loss = log_loss(y_test_encoded, y_test_pred_proba_lr)
lr_precision = precision_score(y_test_encoded, y_test_pred_lr, average='macro')

print(f"Logistic Regression - Accuracy: {lr_accuracy:.4f}, Loss: {lr_loss:.4f}, Precision: {lr_precision:.4f}")

# =============================================================================
# Random Forest 
# =============================================================================

print("\n" + "="*60)
print("Training Model 2: Random Forest")
print("="*60)

depths = range(1, 21)
rf_train_loss_history = []
rf_val_loss_history = []

for depth in depths:
    rf_model = RandomForestClassifier(max_depth=depth, random_state=42)
    rf_model.fit(X_train_split, y_train_split)
    
    y_train_pred_proba = rf_model.predict_proba(X_train_split)
    train_loss = log_loss(y_train_split, y_train_pred_proba)
    rf_train_loss_history.append(train_loss)
    
    y_val_pred_proba = rf_model.predict_proba(X_val_split)
    val_loss = log_loss(y_val_split, y_val_pred_proba)
    rf_val_loss_history.append(val_loss)
    print(f"Depth {depth:2d} → Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

best_depth = depths[np.argmin(rf_val_loss_history)]
print(f"\nBest depth: {best_depth}")

rf_final = RandomForestClassifier(max_depth=best_depth, random_state=42)
rf_final.fit(X_train_pca, y_train_encoded)

y_test_pred_rf = rf_final.predict(X_test_pca)
y_test_pred_proba_rf = rf_final.predict_proba(X_test_pca)

rf_accuracy = accuracy_score(y_test_encoded, y_test_pred_rf)
rf_loss = log_loss(y_test_encoded, y_test_pred_proba_rf)
rf_precision = precision_score(y_test_encoded, y_test_pred_rf, average='macro')

print(f"Random Forest - Accuracy: {rf_accuracy:.4f}, Loss: {rf_loss:.4f}, Precision: {rf_precision:.4f}")

# =============================================================================
# LSTM 
# =============================================================================

print("\n" + "="*60)
print("Training Model 3: LSTM (on raw sensor data)")
print("="*60)

timesteps = X_train_lstm.shape[1]
features = X_train_lstm.shape[2]
num_classes = len(np.unique(y_train_lstm))

lstm_model = Sequential([
    LSTM(128, input_shape=(timesteps, features)),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

lstm_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

lstm_history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_val_lstm, y_val_lstm),
    epochs=30,
    batch_size=64,
    shuffle=True,
    verbose=1
)

lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_raw, y_test_raw)

y_pred_probs_lstm = lstm_model.predict(X_test_raw)
y_pred_lstm = np.argmax(y_pred_probs_lstm, axis=1)

lstm_precision = precision_score(y_test_raw, y_pred_lstm, average='macro')

print(f"\nLSTM - Accuracy: {lstm_accuracy:.4f}, Loss: {lstm_loss:.4f}, Precision: {lstm_precision:.4f}")

# =============================================================================
# COMPARISON TABLE
# =============================================================================

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

print(f"\n{'Model':<25} {'Data Type':<20} {'Accuracy':<12} {'Loss':<12} {'Precision':<12}")
print("-" * 85)
print(f"{'Logistic Regression':<25} {'Pre-extracted (PCA)':<20} {lr_accuracy:<12.4f} {lr_loss:<12.4f} {lr_precision:<12.4f}")
print(f"{'Random Forest':<25} {'Pre-extracted (PCA)':<20} {rf_accuracy:<12.4f} {rf_loss:<12.4f} {rf_precision:<12.4f}")
print(f"{'LSTM':<25} {'Raw Inertial':<20} {lstm_accuracy:<12.4f} {lstm_loss:<12.4f} {lstm_precision:<12.4f}")

# Determine best model
accuracies = {'Logistic Regression': lr_accuracy, 'Random Forest': rf_accuracy, 'LSTM': lstm_accuracy}
best_model = max(accuracies, key=accuracies.get)

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print(f"\n• Best Performing Model: {best_model} with accuracy of {accuracies[best_model]:.4f}")
print(f"\n• Classical ML (Logistic Regression & Random Forest) used pre-extracted features")
print(f"  with PCA dimensionality reduction ({pca.n_components_} components, 95% variance).")
print(f"\n• LSTM was trained on raw inertial sensor data (9 channels × 128 timesteps),")
print(f"  learning temporal patterns directly from the raw signals.")
print(f"\n• Trade-offs:")
print(f"  - Classical ML: Faster training, requires feature engineering")
print(f"  - LSTM: Slower training, learns features automatically from raw data")

# =============================================================================
# GRAPHS
# =============================================================================


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Graph 1: Logistic Regression
axes[0].plot(range(1, n_epochs + 1), lr_train_loss_history, label='Train Loss', color='blue')
axes[0].plot(range(1, n_epochs + 1), lr_val_loss_history, label='Validation Loss', color='orange')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Log Loss')
axes[0].set_title(f'Logistic Regression (lr={lr})')
axes[0].legend()
axes[0].grid(True)
# Graph 2: Random Forest
axes[1].plot(depths, rf_train_loss_history, marker='o', label='Train Loss', color='blue')
axes[1].plot(depths, rf_val_loss_history, marker='s', label='Validation Loss', color='orange')
axes[1].axvline(x=best_depth, color='red', linestyle='--', label=f'Best Depth={best_depth}')
axes[1].set_xlabel('Tree Depth')
axes[1].set_ylabel('Log Loss')
axes[1].set_title('Random Forest - Complexity Curve')
axes[1].legend()
axes[1].grid(True)
# Graph 3: LSTM
axes[2].plot(lstm_history.history['loss'], label='Train Loss', color='blue')
axes[2].plot(lstm_history.history['val_loss'], label='Validation Loss', color='orange')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].set_title('LSTM (Raw Sensor Data)')
axes[2].legend()
axes[2].grid(True)
plt.suptitle('Model Comparison: Training vs Validation Loss', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/model_loss_comparison.png')
plt.show()
plt.close()

# =============================================================================
# BAR CHART COMPARISON
# =============================================================================


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
models = ['Logistic\nRegression', 'Random\nForest', 'LSTM']
colors = ['#3498db', '#2ecc71', '#e74c3c']
# Accuracy comparison
accuracies_list = [lr_accuracy, rf_accuracy, lstm_accuracy]
axes[0].bar(models, accuracies_list, color=colors)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy Comparison')
axes[0].set_ylim(0, 1)
for i, v in enumerate(accuracies_list):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
# Loss comparison
losses_list = [lr_loss, rf_loss, lstm_loss]
axes[1].bar(models, losses_list, color=colors)
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss Comparison')
for i, v in enumerate(losses_list):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
# Precision comparison
precisions_list = [lr_precision, rf_precision, lstm_precision]
axes[2].bar(models, precisions_list, color=colors)
axes[2].set_ylabel('Precision (Macro)')
axes[2].set_title('Precision Comparison')
axes[2].set_ylim(0, 1)
for i, v in enumerate(precisions_list):
    axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
plt.suptitle('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/model_metrics_comparison.png')
plt.show()
plt.close()
