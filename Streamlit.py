import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

ACTIVITY_LABELS = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}

# Train both models once and cache them
@st.cache_resource
def train_models():
    # Load data for Logistic Regression
    X_train_lr = pd.read_csv('dataset/train/X_train.txt', sep=r'\s+', header=None).values
    X_test_lr = pd.read_csv('dataset/test/X_test.txt', sep=r'\s+', header=None).values
    y_train = pd.read_csv('dataset/train/y_train.txt', sep=r'\s+', header=None).values.ravel()
    y_test = pd.read_csv('dataset/test/y_test.txt', sep=r'\s+', header=None).values.ravel()

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Scale and PCA for logistic regression
    scaler_lr = StandardScaler()
    X_train_scaled = scaler_lr.fit_transform(X_train_lr)
    
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Logistic Regression
    lr_model = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.01, random_state=42, max_iter=100)
    lr_model.fit(X_train_pca, y_train_encoded)

    # Load raw sensor data for LSTM
    signals = ['body_acc_x', 'body_acc_y', 'body_acc_z', 
               'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
               'total_acc_x', 'total_acc_y', 'total_acc_z']
    
    train_signals = []
    for signal in signals:
        data = pd.read_csv(f'dataset/train/Inertial Signals/{signal}_train.txt', sep=r'\s+', header=None)
        train_signals.append(data.values)
    
    if not train_signals:
        st.error("Failed to load training signals. Check file paths.")
        return None
    
    X_train_lstm = np.stack(train_signals, axis=2)
    y_train_lstm = y_train - 1  # 0-indexed

    # Scale LSTM data
    scaler_lstm = StandardScaler()
    X_train_reshaped = X_train_lstm.reshape(-1, X_train_lstm.shape[2])
    X_train_scaled_lstm = scaler_lstm.fit_transform(X_train_reshaped)
    X_train_lstm_final = X_train_scaled_lstm.reshape(X_train_lstm.shape)

    # Build and train LSTM
    timesteps = X_train_lstm_final.shape[1]
    features = X_train_lstm_final.shape[2]
    num_classes = 6

    lstm_model = Sequential([
        LSTM(128, input_shape=(timesteps, features)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm_final, y_train_lstm, epochs=10, batch_size=64, verbose=0)

    return {
        'lr_model': lr_model,
        'lstm_model': lstm_model,
        'scaler_lr': scaler_lr,
        'scaler_lstm': scaler_lstm,
        'pca': pca,
        'label_encoder': label_encoder
    }

def predict_lr(features, model_data):
    # features should be shape (561,) or (n, 561)
    features = np.array(features).reshape(1, -1) if len(np.array(features).shape) == 1 else features
    scaled = model_data['scaler_lr'].transform(features)
    pca_features = model_data['pca'].transform(scaled)
    pred = model_data['lr_model'].predict(pca_features)
    proba = model_data['lr_model'].predict_proba(pca_features)
    return pred[0], proba[0]

def predict_lstm(raw_data, model_data):
    # raw_data should be shape (128, 9) for single sample
    raw_data = np.array(raw_data)
    if len(raw_data.shape) == 2:
        raw_data = raw_data.reshape(1, raw_data.shape[0], raw_data.shape[1])
    
    # Scale
    reshaped = raw_data.reshape(-1, raw_data.shape[2])
    scaled = model_data['scaler_lstm'].transform(reshaped)
    scaled = scaled.reshape(raw_data.shape)
    
    proba = model_data['lstm_model'].predict(scaled, verbose=0)
    pred = np.argmax(proba, axis=1)
    return pred[0], proba[0]

def generate_random_lr_features():
    return np.random.randn(561) * 0.5

def generate_random_lstm_data():
    return np.random.randn(128, 9) * 0.3

# Helper to show probabilities with fixed axis
def show_probabilities(proba):
    df = pd.DataFrame({
        'Activity': list(ACTIVITY_LABELS.values()),
        'Probability': proba
    })
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Activity', sort=list(ACTIVITY_LABELS.values()), axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Probability', scale=alt.Scale(domain=[0, 1]))
    ).properties(height=300).configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)

# Main app
def main():
    st.set_page_config(page_title="Activity Recognition", layout="wide")
    st.title("Human Activity Recognition")

    with st.spinner("Training models..."):
        model_data = train_models()

    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "LSTM"]
    )

    input_method = st.radio(
        "Input Method",
        ["Upload CSV", "Random Simulation", "Manual Sliders"],
        horizontal=True
    )

    st.markdown("---")

    if model_choice == "Logistic Regression":
        if input_method == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV with 561 features", type=['csv'])
            if uploaded:
                df = pd.read_csv(uploaded, header=None)
                st.write(f"Loaded {len(df)} samples")
                if st.button("Predict"):
                    for i, row in df.iterrows():
                        pred, proba = predict_lr(row.values, model_data)
                        st.write(f"**Sample {i+1}: {ACTIVITY_LABELS[pred]}**")
                        show_probabilities(proba)
        
        elif input_method == "Random Simulation":
            if st.button("Generate Random Sample & Predict"):
                features = generate_random_lr_features()
                pred, proba = predict_lr(features, model_data)
                st.metric("Predicted Activity", ACTIVITY_LABELS[pred])
                show_probabilities(proba)
                with st.expander("Generated Features (first 20)"):
                    st.write(features[:20])
        
        else:  # Manual Sliders
            features = np.zeros(561)
            cols = st.columns(5)
            feature_names = ['tBodyAcc-mean-X', 'tBodyAcc-mean-Y', 'tBodyAcc-mean-Z',
                           'tBodyAcc-std-X', 'tBodyAcc-std-Y', 'tBodyAcc-std-Z',
                           'tGravityAcc-mean-X', 'tGravityAcc-mean-Y', 'tGravityAcc-mean-Z',
                           'tBodyGyro-mean-X']
            
            for i, name in enumerate(feature_names):
                with cols[i % 5]:
                    features[i] = st.slider(name, -3.0, 3.0, 0.0, 0.1)
            
            if st.button("Predict"):
                pred, proba = predict_lr(features, model_data)
                st.metric("Predicted Activity", ACTIVITY_LABELS[pred])
                show_probabilities(proba)

    else:  # LSTM model
        if input_method == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV (128 rows, 9 columns)", type=['csv'])
            if uploaded:
                df = pd.read_csv(uploaded, header=None)
                st.write(f"Loaded data shape: {df.shape}")
                if df.shape == (128, 9):
                    if st.button("Predict"):
                        pred, proba = predict_lstm(df.values, model_data)
                        st.metric("Predicted Activity", ACTIVITY_LABELS[pred])
                        show_probabilities(proba)
                else:
                    st.error("CSV must have shape (128, 9)")
        
        elif input_method == "Random Simulation":
            if st.button("Generate Random Sensor Data & Predict"):
                raw_data = generate_random_lstm_data()
                pred, proba = predict_lstm(raw_data, model_data)
                st.metric("Predicted Activity", ACTIVITY_LABELS[pred])
                show_probabilities(proba)
                with st.expander("Generated Sensor Data (first 10 timesteps)"):
                    st.dataframe(pd.DataFrame(raw_data[:10], 
                        columns=['body_acc_x', 'body_acc_y', 'body_acc_z',
                                'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                                'total_acc_x', 'total_acc_y', 'total_acc_z']))
        
        else:  # Manual Sliders
            channels = ['body_acc_x', 'body_acc_y', 'body_acc_z',
                       'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                       'total_acc_x', 'total_acc_y', 'total_acc_z']
            
            values = []
            cols = st.columns(3)
            for i, ch in enumerate(channels):
                with cols[i % 3]:
                    val = st.slider(ch, -2.0, 2.0, 0.0, 0.1)
                    values.append(val)
            
            if st.button("Predict"):
                raw_data = np.array(values) + np.random.randn(128, 9) * 0.1
                pred, proba = predict_lstm(raw_data, model_data)
                st.metric("Predicted Activity", ACTIVITY_LABELS[pred])
                show_probabilities(proba)

if __name__ == "__main__":
    main()
