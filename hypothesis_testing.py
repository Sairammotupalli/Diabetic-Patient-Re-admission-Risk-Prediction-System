import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class HypothesisTester:
    def __init__(self):
        self.results = {}
        
    def load_data(self):
        """Load the processed data"""
        print("Loading data for hypothesis testing...")
        
        # Load processed data
        df = pd.read_csv('data/processed_data.csv')
        
        # Load original data for comparison
        original_df = pd.read_csv('data/diabetic_data.csv')
        original_df = original_df.replace('?', np.nan)
        
        return df, original_df
    
    def test_insulin_readmission_correlation(self, df):
        """Test hypothesis: Insulin usage correlates with higher readmission"""
        print("\n=== Testing Insulin Usage vs Readmission Correlation ===")
        
        # Create contingency table
        contingency_table = pd.crosstab(df['insulin_usage'], df['readmitted'])
        print("Contingency Table:")
        print(contingency_table)
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nChi-square test results:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        
        # Fisher's exact test for small sample sizes
        if contingency_table.values.min() < 5:
            print("\nFisher's exact test (recommended for small samples):")
            odds_ratio, fisher_p = fisher_exact(contingency_table)
            print(f"Odds ratio: {odds_ratio:.4f}")
            print(f"p-value: {fisher_p:.4f}")
        
        # Calculate readmission rates
        insulin_readmission_rate = df[df['insulin_usage'] == 1]['readmitted'].mean()
        no_insulin_readmission_rate = df[df['insulin_usage'] == 0]['readmitted'].mean()
        
        print(f"\nReadmission rates:")
        print(f"With insulin: {insulin_readmission_rate:.4f} ({insulin_readmission_rate*100:.2f}%)")
        print(f"Without insulin: {no_insulin_readmission_rate:.4f} ({no_insulin_readmission_rate*100:.2f}%)")
        print(f"Difference: {insulin_readmission_rate - no_insulin_readmission_rate:.4f}")
        
        # Store results
        self.results['insulin_test'] = {
            'chi2': chi2,
            'p_value': p_value,
            'insulin_readmission_rate': insulin_readmission_rate,
            'no_insulin_readmission_rate': no_insulin_readmission_rate,
            'contingency_table': contingency_table
        }
        
        return p_value < 0.05
    
    def test_age_readmission_correlation(self, df):
        """Test correlation between age and readmission using ANOVA"""
        print("\n=== Testing Age vs Readmission Correlation ===")
        
        # Extract age groups from age column
        age_groups = df['age'].unique()
        readmission_by_age = []
        
        for age_group in age_groups:
            readmission_rate = df[df['age'] == age_group]['readmitted'].mean()
            readmission_by_age.append(readmission_rate)
            print(f"Age group {age_group}: {readmission_rate:.4f} ({readmission_rate*100:.2f}%)")
        
        # ANOVA test
        age_data = [df[df['age'] == age]['readmitted'].values for age in age_groups]
        f_stat, p_value = f_oneway(*age_data)
        
        print(f"\nANOVA test results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        self.results['age_test'] = {
            'f_stat': f_stat,
            'p_value': p_value,
            'age_groups': age_groups,
            'readmission_by_age': readmission_by_age
        }
        
        return p_value < 0.05
    
    def test_gender_readmission_correlation(self, df):
        """Test correlation between gender and readmission"""
        print("\n=== Testing Gender vs Readmission Correlation ===")
        
        # Create contingency table
        contingency_table = pd.crosstab(df['gender'], df['readmitted'])
        print("Contingency Table:")
        print(contingency_table)
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nChi-square test results:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        # Calculate readmission rates by gender
        gender_readmission_rates = df.groupby('gender')['readmitted'].mean()
        print(f"\nReadmission rates by gender:")
        for gender, rate in gender_readmission_rates.items():
            print(f"{gender}: {rate:.4f} ({rate*100:.2f}%)")
        
        self.results['gender_test'] = {
            'chi2': chi2,
            'p_value': p_value,
            'gender_readmission_rates': gender_readmission_rates,
            'contingency_table': contingency_table
        }
        
        return p_value < 0.05
    
    def test_medication_count_correlation(self, df):
        """Test correlation between number of medications and readmission"""
        print("\n=== Testing Medication Count vs Readmission Correlation ===")
        
        # Group by medication count and calculate readmission rates
        med_count_readmission = df.groupby('total_medications')['readmitted'].agg(['mean', 'count'])
        print("Readmission rates by medication count:")
        print(med_count_readmission)
        
        # ANOVA test
        med_counts = df['total_medications'].unique()
        med_data = [df[df['total_medications'] == count]['readmitted'].values for count in med_counts]
        f_stat, p_value = f_oneway(*med_data)
        
        print(f"\nANOVA test results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        self.results['medication_test'] = {
            'f_stat': f_stat,
            'p_value': p_value,
            'med_count_readmission': med_count_readmission
        }
        
        return p_value < 0.05
    
    def create_visualizations(self, df):
        """Create visualizations for hypothesis testing"""
        print("\nCreating hypothesis testing visualizations...")
        
        os.makedirs('plots', exist_ok=True)
        
        # Insulin usage vs readmission
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        insulin_readmission = df.groupby('insulin_usage')['readmitted'].mean()
        insulin_readmission.plot(kind='bar', color=['lightblue', 'orange'])
        plt.title('Readmission Rate by Insulin Usage')
        plt.ylabel('Readmission Rate')
        plt.xticks([0, 1], ['No Insulin', 'Insulin'])
        
        # Age vs readmission
        plt.subplot(2, 2, 2)
        age_readmission = df.groupby('age')['readmitted'].mean()
        age_readmission.plot(kind='bar', color='skyblue')
        plt.title('Readmission Rate by Age Group')
        plt.ylabel('Readmission Rate')
        plt.xticks(rotation=45)
        
        # Gender vs readmission
        plt.subplot(2, 2, 3)
        gender_readmission = df.groupby('gender')['readmitted'].mean()
        gender_readmission.plot(kind='bar', color=['pink', 'lightblue'])
        plt.title('Readmission Rate by Gender')
        plt.ylabel('Readmission Rate')
        plt.xticks([0, 1], ['Female', 'Male'])
        
        # Medication count vs readmission
        plt.subplot(2, 2, 4)
        med_readmission = df.groupby('total_medications')['readmitted'].mean()
        med_readmission.plot(kind='line', marker='o', color='green')
        plt.title('Readmission Rate by Medication Count')
        plt.ylabel('Readmission Rate')
        plt.xlabel('Number of Medications')
        
        plt.tight_layout()
        plt.savefig('plots/hypothesis_testing.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Hypothesis testing visualizations saved to plots/hypothesis_testing.png")
    
    def run_hypothesis_tests(self):
        """Run all hypothesis tests"""
        print("Starting hypothesis testing...")
        
        # Load data
        df, original_df = self.load_data()
        
        # Run hypothesis tests
        insulin_significant = self.test_insulin_readmission_correlation(df)
        age_significant = self.test_age_readmission_correlation(df)
        gender_significant = self.test_gender_readmission_correlation(df)
        medication_significant = self.test_medication_count_correlation(df)
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Summary
        print("\n=== Hypothesis Testing Summary ===")
        print(f"Insulin usage correlation: {'Significant' if insulin_significant else 'Not significant'}")
        print(f"Age correlation: {'Significant' if age_significant else 'Not significant'}")
        print(f"Gender correlation: {'Significant' if gender_significant else 'Not significant'}")
        print(f"Medication count correlation: {'Significant' if medication_significant else 'Not significant'}")
        
        # Save results
        joblib.dump(self.results, 'models/hypothesis_test_results.pkl')
        print("\nHypothesis test results saved to models/hypothesis_test_results.pkl")
        
        return self.results

if __name__ == "__main__":
    import os
    tester = HypothesisTester()
    results = tester.run_hypothesis_tests() 