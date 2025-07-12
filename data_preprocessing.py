import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='most_frequent')
        
    def load_data(self, filepath):
        """Load the diabetic dataset"""
        print("Loading data...")
        df = pd.read_csv(filepath)
        print(f"Original data shape: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and dropping irrelevant columns"""
        print("Cleaning data...")
        
        # Replace '?' with NaN
        df = df.replace('?', np.nan)
        
        # Drop high-missing or irrelevant columns
        columns_to_drop = ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"]
        df = df.drop(columns_to_drop, axis=1, errors='ignore')
        
        # Handle missing values
        categorical_cols = df.select_dtypes(include=['object']).columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Impute categorical columns with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        # Impute numerical columns with median
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        print(f"Data shape after cleaning: {df.shape}")
        return df
    
    def consolidate_categories(self, df):
        """Consolidate medication and diagnosis categories"""
        print("Consolidating categories...")
        
        # Medication columns (all columns that contain medication names)
        medication_cols = [col for col in df.columns if any(med in col.lower() for med in 
                          ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                           'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                           'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                           'miglitol', 'troglitazone', 'tolazamide', 'examide', 
                           'citoglipton', 'insulin'])]
        
        # Create consolidated medication features
        df['insulin_usage'] = (df['insulin'] == 'Up').astype(int)
        df['oral_medications'] = df[medication_cols].apply(lambda x: (x == 'Up').sum(), axis=1)
        df['total_medications'] = df[medication_cols].apply(lambda x: (x != 'No').sum(), axis=1)
        
        # Diagnosis consolidation
        diagnosis_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diagnosis_cols:
            if col in df.columns:
                # Create binary features for common diagnosis categories
                df[f'{col}_diabetes'] = df[col].str.contains('250', na=False).astype(int)
                df[f'{col}_circulatory'] = df[col].str.contains('^[4-5]', na=False).astype(int)
                df[f'{col}_respiratory'] = df[col].str.contains('^[4-5]', na=False).astype(int)
        
        print(f"Data shape after consolidation: {df.shape}")
        return df
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables using label encoding"""
        print("Encoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'readmitted':  # Don't encode target yet
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Encode target variable
        df['readmitted'] = df['readmitted'].map({'NO': 0, '<30': 1, '>30': 0})
        
        print(f"Data shape after encoding: {df.shape}")
        return df
    
    def balance_dataset(self, X, y):
        """Balance the dataset using SMOTE"""
        print("Balancing dataset with SMOTE...")
        print(f"Original class distribution: {np.bincount(y)}")
        
        # Handle numerical issues before SMOTE
        X_clean = X.copy()
        
        # Replace infinite values with NaN, then fill with median
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for each column
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'int64']:
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_clean[col] = X_clean[col].fillna(median_val)
        
        # Ensure all values are finite
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        # Clip extreme values to prevent overflow
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'int64']:
                # Clip to 3 standard deviations
                mean_val = X_clean[col].mean()
                std_val = X_clean[col].std()
                if std_val > 0:
                    X_clean[col] = X_clean[col].clip(
                        mean_val - 3 * std_val, 
                        mean_val + 3 * std_val
                    )
        
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X_clean, y)
            
            print(f"Balanced class distribution: {np.bincount(y_balanced)}")
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"SMOTE failed: {e}")
            print("Using original dataset without balancing...")
            return X_clean, y
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        # Separate features and target
        X = df.drop('readmitted', axis=1)
        y = df['readmitted']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Balance training data
        X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train)
        
        # Handle numerical issues before scaling
        X_train_clean = X_train_balanced.copy()
        X_test_clean = X_test.copy()
        
        # Replace infinite values
        X_train_clean = X_train_clean.replace([np.inf, -np.inf], np.nan)
        X_test_clean = X_test_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        for col in X_train_clean.columns:
            if X_train_clean[col].dtype in ['float64', 'int64']:
                median_val = X_train_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_train_clean[col] = X_train_clean[col].fillna(median_val)
                X_test_clean[col] = X_test_clean[col].fillna(median_val)
        
        # Scale numerical features with robust scaling
        numerical_cols = X_train_clean.select_dtypes(include=['int64', 'float64']).columns
        
        try:
            # Use RobustScaler instead of StandardScaler to handle outliers
            from sklearn.preprocessing import RobustScaler
            robust_scaler = RobustScaler()
            
            X_train_clean[numerical_cols] = robust_scaler.fit_transform(X_train_clean[numerical_cols])
            X_test_clean[numerical_cols] = robust_scaler.transform(X_test_clean[numerical_cols])
            
            # Update the scaler attribute
            self.scaler = robust_scaler
            
        except Exception as e:
            print(f"Scaling failed: {e}")
            print("Using unscaled features...")
            # Keep original values if scaling fails
        
        return X_train_clean, X_test_clean, y_train_balanced, y_test
    
    def save_preprocessing_artifacts(self):
        """Save preprocessing artifacts for later use"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.imputer, 'models/imputer.pkl')
        print("Preprocessing artifacts saved to models/")
    
    def run_preprocessing_pipeline(self, input_file, output_file):
        """Run the complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        # Load data
        df = self.load_data(input_file)
        
        # Clean data
        df = self.clean_data(df)
        
        # Consolidate categories
        df = self.consolidate_categories(df)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df)
        
        # Prepare features
        X_train, X_test, y_train, y_test = self.prepare_features(df)
        
        # Save processed data
        os.makedirs('data', exist_ok=True)
        df.to_csv(output_file, index=False)
        
        # Save train/test splits
        joblib.dump(X_train, 'models/X_train.pkl')
        joblib.dump(X_test, 'models/X_test.pkl')
        joblib.dump(y_train, 'models/y_train.pkl')
        joblib.dump(y_test, 'models/y_test.pkl')
        
        # Save preprocessing artifacts
        self.save_preprocessing_artifacts()
        
        print(f"Preprocessing complete! Processed data saved to {output_file}")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df, X_train, X_test, y_train, y_test = preprocessor.run_preprocessing_pipeline(
        'data/diabetic_data.csv',
        'data/processed_data.csv'
    ) 