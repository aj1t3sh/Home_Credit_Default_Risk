"""
Exploratory Data Analysis (EDA) Script for Home Credit Default Risk.

Generates comprehensive EDA statistics and saves key figures into reports/figures/:
1. target_distribution.png
2. missing_values.png
3. ext_sources_kde.png
4. age_vs_default.png
5. correlation_matrix.png
6. days_employed_anomaly.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import get_data_path, timer, get_logger

logger = get_logger("eda_analysis")


def run_eda(project_root: str = None) -> dict:
    """
    Run exploratory data analysis and generate visualization artifacts.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    fig_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})
    
    with timer("Running Exploratory Data Analysis", logger):
        train_path = get_data_path("application_train.csv", project_root)
        test_path = get_data_path("application_test.csv", project_root)
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
        
        # 1. Target Distribution Plot
        plt.figure(figsize=(8, 5))
        counts = train_df['TARGET'].value_counts()
        percs = train_df['TARGET'].value_counts(normalize=True) * 100
        bars = plt.bar(['Non-Default (0)', 'Default (1)'], counts.values, color=['#2b5c8f', '#d95f02'], width=0.55, edgecolor='black', alpha=0.85)
        
        for bar, count, perc in zip(bars, counts.values, percs.values):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5000, f"{count:,}\n({perc:.2f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.title('Target Variable Distribution (Loan Default)', fontsize=14, fontweight='bold', pad=12)
        plt.ylabel('Number of Applicants', fontsize=12, fontweight='bold')
        plt.ylim(0, max(counts.values) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "target_distribution.png"), dpi=300)
        plt.close()
        logger.info("Saved target_distribution.png")
        
        # 2. Missing Values Analysis
        missing = train_df.isnull().sum()
        missing_perc = (missing / len(train_df)) * 100
        missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Perc': missing_perc}).query('Missing_Count > 0').sort_values('Missing_Perc', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_missing = missing_df.head(30)
        sns.barplot(x=top_missing['Missing_Perc'], y=top_missing.index, palette='rocket')
        plt.title('Top 30 Features by Missing Value Percentage in application_train', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Feature Name', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "missing_values.png"), dpi=300)
        plt.close()
        logger.info("Saved missing_values.png")
        
        # 3. EXT_SOURCE Distributions (KDE)
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for idx, col in enumerate(ext_cols):
            sns.kdeplot(train_df.loc[train_df['TARGET'] == 0, col], label='Non-Default (0)', color='#1f77b4', fill=True, alpha=0.3, ax=axes[idx], lw=2)
            sns.kdeplot(train_df.loc[train_df['TARGET'] == 1, col], label='Default (1)', color='#d62728', fill=True, alpha=0.3, ax=axes[idx], lw=2)
            axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Normalized External Score', fontsize=11)
            axes[idx].set_ylabel('Density', fontsize=11)
            axes[idx].legend(loc='upper right', frameon=True)
            axes[idx].grid(True, linestyle=':', alpha=0.6)
            
        plt.suptitle('External Source Normalized Scores vs Loan Default', fontsize=15, fontweight='bold', y=1.03)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "ext_sources_kde.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved ext_sources_kde.png")
        
        # 4. Age vs Default Rate
        train_df['AGE_YEARS'] = -train_df['DAYS_BIRTH'] / 365.25
        train_df['AGE_GROUP'] = pd.cut(train_df['AGE_YEARS'], bins=np.arange(20, 75, 5))
        age_group_df = train_df.groupby('AGE_GROUP', observed=True)['TARGET'].agg(['count', 'mean']).reset_index()
        age_group_df['Default_Rate_Perc'] = age_group_df['mean'] * 100
        
        plt.figure(figsize=(10, 5.5))
        bar_plot = sns.barplot(
            x='AGE_GROUP',
            y='Default_Rate_Perc',
            data=age_group_df,
            palette='Blues_d',
            edgecolor='black'
        )
        for p in bar_plot.patches:
            bar_plot.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 4),
                              textcoords='offset points')
        plt.title('Default Rate by Age Group (Younger Applicants Show Higher Risk)', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Age Group (Years)', fontsize=12, fontweight='bold')
        plt.ylabel('Default Rate (%)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "age_vs_default.png"), dpi=300)
        plt.close()
        logger.info("Saved age_vs_default.png")
        
        # 5. Correlation Heatmap for Key Numerical Variables
        key_num_cols = [
            'TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL',
            'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH',
            'DAYS_REGISTRATION', 'REGION_RATING_CLIENT'
        ]
        available_cols = [c for c in key_num_cols if c in train_df.columns]
        corr_matrix = train_df[available_cols].corr()
        
        plt.figure(figsize=(11, 9))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Correlation Matrix of Key Numerical Variables and Target', fontsize=14, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "correlation_matrix.png"), dpi=300)
        plt.close()
        logger.info("Saved correlation_matrix.png")
        
        # 6. Days Employed Anomaly Analysis
        plt.figure(figsize=(10, 5))
        anom_counts = (train_df['DAYS_EMPLOYED'] == 365243).value_counts()
        plt.bar(['Normal Records', 'Anomalous (365,243 Days)'], anom_counts.values, color=['#3470a3', '#c93b2b'], width=0.5, edgecolor='black')
        for idx, (val, count) in enumerate(anom_counts.items()):
            plt.text(idx, count + 4000, f"{count:,}\n({(count/len(train_df))*100:.2f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.title('DAYS_EMPLOYED: Identification of 365,243 Anomaly (18.01% of Data)', fontsize=13, fontweight='bold', pad=12)
        plt.ylabel('Applicant Count', fontsize=12, fontweight='bold')
        plt.ylim(0, max(anom_counts.values) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "days_employed_anomaly.png"), dpi=300)
        plt.close()
        logger.info("Saved days_employed_anomaly.png")
        
        eda_summary = {
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'train_cols': len(train_df.columns),
            'test_cols': len(test_df.columns),
            'default_rate': train_df['TARGET'].mean(),
            'top_missing_cols': missing_df.head(10).to_dict()
        }
        
    return eda_summary


if __name__ == "__main__":
    run_eda()
