import streamlit as st
import numpy as np
import pandas as pd
from obspy import read
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import os

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={df.columns[0]: 'time_abs(%Y-%m-%dT%H:%M:%S.%f)', df.columns[1]: 'time_rel(sec)', df.columns[2]: 'velocity(m/s)'}, inplace=True)
    df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'] = pd.to_datetime(df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
    df['time_rel(sec)'] = pd.to_numeric(df['time_rel(sec)'], errors='coerce')
    df['velocity(m/s)'] = pd.to_numeric(df['velocity(m/s)'], errors='coerce')
    df.dropna(inplace=True)
    return df

# Function to generate waveform
def generate_waveform(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time_rel(sec)'], df['velocity(m/s)'], label='Velocity Waveform', color='blue')
    plt.title('Velocity Waveform')
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (m/s)')
    plt.grid()
    plt.legend()
    st.pyplot(plt)

# Function to create a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the bandpass filter
def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    required_length = 35
    if len(data) < required_length:
        pad_width = ((required_length - len(data)) // 2, (required_length - len(data)) // 2)
        data = np.pad(data, pad_width, 'constant')
        y = filtfilt(b, a, data)
        return y
    else:
        return filtfilt(b, a, data)

# Function to classify signals based on amplitude
def classify_signals(df, lowcut=1.0, highcut=10.0, fs=100.0, amplitude_threshold=1.0):
    results = []
    for index, row in df.iterrows():
        time = row['time_rel(sec)']
        velocity = row['velocity(m/s)']

        filtered_signal = apply_bandpass_filter(np.array([velocity]), lowcut, highcut, fs)
        if len(filtered_signal) > 0 and np.issubdtype(filtered_signal.dtype, np.number):
            filtered_signal = filtered_signal[~np.isnan(filtered_signal)]
            max_amplitude = np.max(filtered_signal)
            min_amplitude = np.min(filtered_signal)
            label = 'valid' if max_amplitude >= amplitude_threshold and min_amplitude <= -amplitude_threshold else 'invalid'
        else:
            label = 'invalid'
        results.append({'index': index, 'time_rel(sec)': time, 'velocity(m/s)': velocity, 'label': label})
    return pd.DataFrame(results)

# Main app logic
def main():
    st.title("Seismic Detection App")

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Load and preprocess the data
        df = load_and_preprocess_data(uploaded_file)

        # Display dataframe
        st.write("Data Preview:", df.head())

        # Generate waveform
        st.subheader("Velocity Waveform")
        generate_waveform(df)

        # Classify signals
        classified_df = classify_signals(df)
        st.write("Classification Results:", classified_df)

        # Prepare data for model training
        classified_df['label'] = classified_df['label'].map({'valid': 1, 'invalid': 0}).astype(float)
        scale = StandardScaler()
        classified_df['velocity(m/s)'] = scale.fit_transform(classified_df['velocity(m/s)'].values.reshape(-1, 1))
        classified_df['time_rel(sec)'] = scale.fit_transform(classified_df['time_rel(sec)'].values.reshape(-1, 1))
        X = classified_df.drop('label', axis=1)
        y = classified_df['label']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Neural network model
        model = keras.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        # Evaluate the model
        evaluation = model.evaluate(X_test, y_test)
        st.write("Model Evaluation:", evaluation)

        # Plot training history
        st.subheader("Training History")
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        st.pyplot(plt)

if __name__ == "__main__":
    main()