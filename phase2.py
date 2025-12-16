import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#--------------------------------------------------
#Phase 2
#--------------------------------------------------

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

y_train = pd.read_csv('dataset/train/y_train.txt', sep=r'\s+', header=None)
y_test = pd.read_csv('dataset/test/y_test.txt', sep=r'\s+', header=None)

X_train = np.stack([
    body_acc_x_train.values,
    body_acc_y_train.values,
    body_acc_z_train.values,
    body_gyro_x_train.values,
    body_gyro_y_train.values,
    body_gyro_z_train.values,
    total_acc_x_train.values,
    total_acc_y_train.values,
    total_acc_z_train.values
], axis=2)

X_test = np.stack([
    body_acc_x_test.values,
    body_acc_y_test.values,
    body_acc_z_test.values,
    body_gyro_x_test.values,
    body_gyro_y_test.values,
    body_gyro_z_test.values,
    total_acc_x_test.values,
    total_acc_y_test.values,
    total_acc_z_test.values
], axis=2)


y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

y_train = y_train - 1
y_test = y_test - 1

activity_labels = pd.read_csv('dataset/activity_labels.txt', sep=' ', header=None, names=['id', 'activity'])
print("Activity Labels:")
print(activity_labels)


print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train))


scaler = StandardScaler()

X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
X_test_reshaped = X_test.reshape(-1, X_test.shape[2])

X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

X_train = X_train_scaled.reshape(X_train.shape)
X_test = X_test_scaled.reshape(X_test.shape)



X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)


timesteps = X_train.shape[1]
features = X_train.shape[2]
num_classes = len(np.unique(y_train))

model = Sequential([
    LSTM(128, input_shape=(timesteps, features)),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    shuffle=True
)

loss, accuracy = model.evaluate(X_test, y_test)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('graphs/LSTM_Loss.png')
plt.show()



y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test loss: {loss:.4f}" )


prec = precision_score(y_test, y_pred, average='macro')
print(f"Test Precision (macro): {prec:.4f}")


