import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import os

# Optimized Firefly Algorithm
class FireflyAlgorithm:
    def __init__(self, objective_function, dimensions, population_size, max_iter, bounds, alpha=0.2, beta0=1.0, gamma=1.0):
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.population_size = population_size
        self.max_iter = max_iter
        self.bounds = bounds
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.convergence = []

    def run(self):
        fireflies = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dimensions))
        intensity = np.array([self.objective_function(f) for f in fireflies])

        best_idx = np.argmin(intensity)
        best_solution = fireflies[best_idx].copy()
        best_fitness = intensity[best_idx]
        self.convergence.append(best_fitness)

        no_improve_count = 0
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                better_mask = intensity[i] > intensity
                if not np.any(better_mask):
                    continue
                distances = np.linalg.norm(fireflies[i] - fireflies[better_mask], axis=1)
                for j_idx, j in enumerate(np.where(better_mask)[0]):
                    beta = self.beta0 * np.exp(-self.gamma * distances[j_idx] ** 2)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + self.alpha * (np.random.rand(self.dimensions) - 0.5)
                fireflies[i] = np.clip(fireflies[i], self.bounds[:, 0], self.bounds[:, 1])
                intensity[i] = self.objective_function(fireflies[i])

            current_best_idx = np.argmin(intensity)
            current_best_fitness = intensity[current_best_idx]
            if current_best_fitness < best_fitness:
                best_solution = fireflies[current_best_idx].copy()
                best_fitness = current_best_fitness
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.convergence.append(best_fitness)

            if no_improve_count >= 5:  # Early stopping
                break

        return best_solution, best_fitness


class StrokePredictionApp:
    def __init__(self):
        self.init_session_state()
        self.load_dataset()
        self.init_encoders()

    def init_session_state(self):
        if 'dataset' not in st.session_state:
            st.session_state.dataset = None
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'encoders_fitted' not in st.session_state:
            st.session_state.encoders_fitted = False
        if 'encoders' not in st.session_state:
            st.session_state.encoders = {
                'gender': LabelEncoder(),
                'ever_married': LabelEncoder(),
                'work_type': LabelEncoder(),
                'Residence_type': LabelEncoder(),
                'smoking_status': LabelEncoder()
            }
        if 'scaler' not in st.session_state:
            st.session_state.scaler = StandardScaler()

    def init_encoders(self):
        if st.session_state.data_loaded and not st.session_state.encoders_fitted:
            self.fit_encoders()

    def fit_encoders(self):
        try:
            df = st.session_state.dataset.copy()
            categorical_cols = {
                'gender': st.session_state.encoders['gender'],
                'ever_married': st.session_state.encoders['ever_married'],
                'work_type': st.session_state.encoders['work_type'],
                'Residence_type': st.session_state.encoders['Residence_type'],
                'smoking_status': st.session_state.encoders['smoking_status']
            }
            
            # Fit all encoders with all possible categories
            for col, encoder in categorical_cols.items():
                if col in df.columns:
                    encoder.fit(df[col].astype(str))
            
            st.session_state.encoders_fitted = True
        except Exception as e:
            st.error(f"Error fitting encoders: {str(e)}")

    def load_dataset(self):
        try:
            dataset_path = self.find_dataset_file()
            if dataset_path is None:
                st.error("Dataset not found in 'balanced_stroke_dataset' folder")
                return False

            if dataset_path.endswith('.csv'):
                st.session_state.dataset = pd.read_csv(dataset_path)
            else:
                st.session_state.dataset = pd.read_excel(dataset_path)

            st.session_state.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return False

    def find_dataset_file(self):
        dataset_folder = "balanced_stroke_dataset"
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder, exist_ok=True)
            return None

        for file in os.listdir(dataset_folder):
            if file.lower().endswith('.csv'):
                return os.path.join(dataset_folder, file)
            elif file.lower().endswith(('.xls', '.xlsx')):
                return os.path.join(dataset_folder, file)
        return None

    def run(self):
        st.title("🧠 Stroke Prediction System")
        st.markdown("Using SVM optimized with Firefly Algorithm")

        if not st.session_state.data_loaded:
            st.warning("Please place your dataset in the 'balanced_stroke_dataset' folder")
            return

        tabs = st.tabs(["📊 Prediction", "📂 Dataset", "⚙️ Training", "🕒 History"])
        with tabs[0]:
            self.prediction_interface()
        with tabs[1]:
            self.dataset_interface()
        with tabs[2]:
            self.training_interface()
        with tabs[3]:
            self.history_interface()

    def preprocess_data(self, test_size=0.2):
        try:
            df = st.session_state.dataset.copy()
            self.handle_missing_values(df)
            if 'id' in df.columns:
                df = df.drop('id', axis=1)

            # Transform using fitted encoders
            categorical_cols = {
                'gender': st.session_state.encoders['gender'],
                'ever_married': st.session_state.encoders['ever_married'],
                'work_type': st.session_state.encoders['work_type'],
                'Residence_type': st.session_state.encoders['Residence_type'],
                'smoking_status': st.session_state.encoders['smoking_status']
            }
            
            for col, encoder in categorical_cols.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col].astype(str))

            X = df.drop('stroke', axis=1)
            y = df['stroke']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            st.session_state.scaler = StandardScaler()
            X_train = st.session_state.scaler.fit_transform(X_train)
            X_test = st.session_state.scaler.transform(X_test)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            return True
        except Exception as e:
            st.error(f"Data preprocessing failed: {str(e)}")
            return False

    def handle_missing_values(self, df):
        missing = df.isnull().sum()
        if missing.sum() == 0:
            return

        numeric_cols = ['age', 'avg_glucose_level', 'bmi']
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)

    def prediction_interface(self):
        st.header("Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender:", ["Female", "Male"])
            age = st.number_input("Age:", min_value=0, max_value=120, value=30)
            hypertension = st.selectbox("Hypertension:", [0, 1])
            heart_disease = st.selectbox("Heart Disease:", [0, 1])
            married = st.selectbox("Ever Married:", ["Yes", "No"])
        with col2:
            work_type = st.selectbox("Work Type:", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence = st.selectbox("Residence Type:", ["Urban", "Rural"])
            glucose = st.number_input("Avg Glucose Level:", min_value=50.0, max_value=300.0, value=100.0)
            bmi = st.number_input("BMI:", min_value=10.0, max_value=50.0, value=25.0)
            smoking = st.selectbox("Smoking Status:", ["never smoked", "formerly smoked", "smokes", "Unknown"])

        if st.button("Predict Stroke Risk"):
            self.predict_stroke({
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': married,
                'work_type': work_type,
                'Residence_type': residence,
                'avg_glucose_level': glucose,
                'bmi': bmi,
                'smoking_status': smoking
            })

        if st.button("Clear Form"):
            st.experimental_rerun()

    def dataset_interface(self):
        st.header("Dataset Management")
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.dataset)
        if st.button("Show Statistics"):
            self.show_statistics()

    def training_interface(self):
        st.header("Model Training")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size (%):", 10, 40, 20)
            population_size = st.number_input("Firefly Population:", min_value=5, max_value=50, value=10)
        with col2:
            max_iterations = st.number_input("Max Iterations:", min_value=10, max_value=100, value=50)

        if st.button("Train Model"):
            if st.session_state.dataset is None:
                st.error("No dataset loaded!")
                return
            
            # Ensure encoders are fitted before training
            if not st.session_state.encoders_fitted:
                self.fit_encoders()
                
            with st.spinner("Training model..."):
                success = self.preprocess_data(test_size / 100.0)
                if success:
                    self.train_model(population_size, max_iterations)

        if st.button("Evaluate Model"):
            if st.session_state.model is None:
                st.error("No trained model available!")
                return
            with st.spinner("Evaluating model..."):
                self.evaluate_model()

    def history_interface(self):
        st.header("Prediction History")
        if not st.session_state.history:
            st.info("No prediction history yet.")
            return
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
        if st.button("Clear History"):
            st.session_state.history = []
            st.experimental_rerun()
        if st.button("Export History"):
            self.export_history(history_df)

    def show_statistics(self):
        with st.expander("Dataset Statistics", expanded=True):
            st.subheader("Basic Information")
            st.write(f"Dataset Shape: {st.session_state.dataset.shape}")
            st.subheader("Missing Values")
            st.write(st.session_state.dataset.isnull().sum())
            st.subheader("Descriptive Statistics")
            st.write(st.session_state.dataset.describe())
            if 'stroke' in st.session_state.dataset.columns:
                st.subheader("Class Distribution")
                st.write(st.session_state.dataset['stroke'].value_counts())

    def firefly_objective(self, params):
        C = max(params[0], 0.1)
        gamma = max(params[1], 0.0001)
        model = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)
        model.fit(st.session_state.X_train, st.session_state.y_train)
        y_pred = model.predict(st.session_state.X_train)
        accuracy = accuracy_score(st.session_state.y_train, y_pred)
        return -accuracy

    def train_model(self, population_size, max_iter):
        try:
            bounds = np.array([[0.1, 100], [0.0001, 10]])
            fa = FireflyAlgorithm(self.firefly_objective, 2, population_size, max_iter, bounds, alpha=0.5, beta0=1.0, gamma=0.01)
            best_params, best_fitness = fa.run()

            st.session_state.model = SVC(C=max(best_params[0], 0.1), gamma=max(best_params[1], 0.0001),
                                         kernel='rbf', probability=True, random_state=42)
            st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)

            st.success("✅ Training completed successfully!")
            st.write(f"**Best Parameters:**\n- C: {best_params[0]:.4f}\n- gamma: {best_params[1]:.4f}")
            st.write(f"**Training Accuracy:** {-best_fitness * 100:.2f}%")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(fa.convergence, 'b-', linewidth=2)
            ax.set_title('Firefly Algorithm Convergence')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Fitness (Negative Accuracy)')
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    def evaluate_model(self):
        try:
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            accuracy = accuracy_score(st.session_state.y_test, y_pred)
            cm = confusion_matrix(st.session_state.y_test, y_pred)
            report = classification_report(st.session_state.y_test, y_pred)

            st.success("Model evaluation completed!")
            st.subheader("Evaluation Results")
            st.write(f"Test Accuracy: {accuracy * 100:.2f}%")

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['No Stroke', 'Stroke'])
            ax.set_yticklabels(['No Stroke', 'Stroke'])
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.text(report)
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")

    def predict_stroke(self, features):
        try:
            if st.session_state.model is None:
                st.error("No trained model available!")
                return
                
            if not st.session_state.encoders_fitted:
                st.error("Encoders not fitted! Please train the model first.")
                return

            # Transform input features using the fitted encoders from session state
            features_encoded = [
                st.session_state.encoders['gender'].transform([features['gender']])[0],
                features['age'],
                features['hypertension'],
                features['heart_disease'],
                st.session_state.encoders['ever_married'].transform([features['ever_married']])[0],
                st.session_state.encoders['work_type'].transform([features['work_type']])[0],
                st.session_state.encoders['Residence_type'].transform([features['Residence_type']])[0],
                features['avg_glucose_level'],
                features['bmi'],
                st.session_state.encoders['smoking_status'].transform([features['smoking_status']])[0]
            ]
            
            features_array = np.array([features_encoded])
            features_scaled = st.session_state.scaler.transform(features_array)
            prediction = st.session_state.model.predict(features_scaled)[0]
            probability = st.session_state.model.predict_proba(features_scaled)[0][1]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"High Risk of Stroke ({probability * 100:.2f}% probability)")
                st.warning("Warning: Consult a healthcare professional immediately.")
            else:
                st.success(f"Low Risk of Stroke ({probability * 100:.2f}% probability)")
                st.info("Maintain a healthy lifestyle with regular check-ups.")

            self.add_to_history(features, prediction, probability)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    def add_to_history(self, features, prediction, probability):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        features_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
        st.session_state.history.append({
            "Timestamp": timestamp,
            "Features": features_str,
            "Prediction": "High Risk" if prediction == 1 else "Low Risk",
            "Probability": f"{probability * 100:.2f}%"
        })

    def export_history(self, history_df):
        try:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download History as CSV",
                data=csv,
                file_name="stroke_prediction_history.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Failed to export history: {str(e)}")


if __name__ == "__main__":
    app = StrokePredictionApp()
    app.run()