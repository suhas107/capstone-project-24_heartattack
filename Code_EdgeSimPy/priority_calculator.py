import numpy as np
import sys
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports (keeps your original project layout)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models import HealthTask, Patient

class PriorityCalculator:
    def __init__(self):
        # Normal ranges for general patients (these were in your original code)
        self.normal_ranges = {
            'heart_rate': (60, 100),      # bpm
            'blood_pressure': (90, 120),  # mmHg
            'glucose_level': (70, 140)    # mg/dL
        }

        # Parameter weights for m-value (urgency hierarchy/tie breaking mechanism)
        self.parameter_weights = {
            'heart_rate': 3.0,
            'blood_pressure': 2.0,
            'glucose_level': 1.0
        }

        # ANN models and scalers
        self.ann_model_k = None
        self.ann_model_m = None
        self.feature_scaler = None
        self.train_ml_models()

    def train_ml_models(self):
        """Train ANN regressors (MLPRegressor) using medical data."""
        try:
            from src.real_data_loader import RealDataLoader
            data_loader = RealDataLoader()
            real_data = data_loader.load_real_training_data()
            if real_data is None or len(real_data) == 0:
                print("PriorityCalculator: No real data returned for training; skipping ANN training.")
                return
            feature_columns = ['heart_rate', 'blood_pressure', 'glucose_level', 'age', 'height', 'weight', 'gender']
            for col in feature_columns:
                if col not in real_data.columns:
                    raise ValueError(f"Expected feature column '{col}' not found in real_data")
            X = real_data[feature_columns].values.astype(float)
            if real_data['gender'].dtype == object:
                gender_map = {'M': 1, 'F': 0, 'm': 1, 'f': 0}
                X[:, -1] = real_data['gender'].map(lambda v: gender_map.get(v, 0)).astype(float)
            if 'k_value' not in real_data.columns or 'm_value' not in real_data.columns:
                raise ValueError("Training data must contain 'k_value' and 'm_value' columns")
            y_k = real_data['k_value'].values.astype(float)
            y_m = real_data['m_value'].values.astype(float)
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            ann_kwargs = {
                'hidden_layer_sizes': (32, 16, 8),
                'activation': 'relu',
                'solver': 'adam',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'random_state': 42,
                'early_stopping': True,
                'n_iter_no_change': 20,
                'verbose': False
            }

            # Create & train model for k and m
            self.ann_model_k = MLPRegressor(**ann_kwargs)
            self.ann_model_m = MLPRegressor(**ann_kwargs)

            print("PriorityCalculator: Training ANN for k_value...")
            self.ann_model_k.fit(X_scaled, y_k)
            print("PriorityCalculator: Training ANN for m_value...")
            self.ann_model_m.fit(X_scaled, y_m)

            print("PriorityCalculator: ANN training complete.")

        except Exception as e:
            print(f"PriorityCalculator: Error during ANN training: {e}")
            self.ann_model_k = None
            self.ann_model_m = None
            self.feature_scaler = None

    def _calculate_parameter_urgency(self, value: float, lower_bound: float, upper_bound: float) -> float:
        """Calculate urgency for a single parameter using the research formula."""
        # |(ub - value)^2 - (lb - value)^2| / (ub - lb)^2
        try:
            denom = (upper_bound - lower_bound) ** 2
            if denom == 0:
                return 0.0
            urgency = abs((upper_bound - value) ** 2 - (lower_bound - value) ** 2) / denom
            return float(min(urgency, 2.0))
        except Exception:
            return 0.0

    def calculate_general_priority(self, task: HealthTask) -> tuple[float, float]:
        """Calculate priority for general patient type using a deterministic rule-based approach."""
        hr_k = self._calculate_parameter_urgency(task.heart_rate, *self.normal_ranges['heart_rate'])
        bp_k = self._calculate_parameter_urgency(task.blood_pressure, *self.normal_ranges['blood_pressure'])
        glucose_k = self._calculate_parameter_urgency(task.glucose_level, *self.normal_ranges['glucose_level'])

        k_value = max(hr_k, bp_k, glucose_k)

        urgencies = {
            'heart_rate': hr_k,
            'blood_pressure': bp_k,
            'glucose_level': glucose_k
        }
        most_critical = max(urgencies, key=urgencies.get)
        m_value = self.parameter_weights.get(most_critical, 1.0)

        return k_value, m_value

    def calculate_specific_priority(self, task: HealthTask, patient: Patient) -> tuple[float, float]:
        """Calculate priority for a specific patient type using trained ANN models."""
        # If ANN not trained, fall back to rule-based
        if self.ann_model_k is None or self.ann_model_m is None or self.feature_scaler is None:
            # fallback
            return self.calculate_general_priority(task)

        # Prepare features for ANN (match training order)
        gender_val = 1 if (patient.gender == 'M' or patient.gender == 'm' or patient.gender == 1) else 0
        features = np.array([[
            float(task.heart_rate),
            float(task.blood_pressure),
            float(task.glucose_level),
            float(patient.age),
            float(patient.height),
            float(patient.weight),
            float(gender_val)
        ]])

        # Scale features
        try:
            features_scaled = self.feature_scaler.transform(features)
        except Exception:
            # Fallback to raw if scaler fails
            features_scaled = features

        # Predict using ANN regressors
        try:
            k_value = float(self.ann_model_k.predict(features_scaled)[0])
            m_value = float(self.ann_model_m.predict(features_scaled)[0])
        except Exception as e:
            # If prediction fails, fallback to rule-based
            print(f"PriorityCalculator: ANN prediction error: {e}; falling back to rule-based.")
            return self.calculate_general_priority(task)

        # Clamp k_value to [0, 2]
        k_value = max(0.0, min(k_value, 2.0))

        # Optionally round or map m_value to expected discrete set (1.0,2.0,3.0)
        # If your m_value is expected to be one of [1.0, 2.0, 3.0], you can map:
        # m_value = float(min([1.0,2.0,3.0], key=lambda x: abs(x - m_value)))
        # For now, we keep predicted continuous m_value (you can choose to round)
        return k_value, m_value

    def calculate_task_priority(self, task: HealthTask, patient: Patient) -> HealthTask:
        """Calculate and attach priority values for a task based on patient type."""
        if task.task_type == 'general':
            k_value, m_value = self.calculate_general_priority(task)
        else:
            k_value, m_value = self.calculate_specific_priority(task, patient)

        task.k_value = k_value
        task.m_value = m_value
        return task
