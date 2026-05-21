import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class EDA:

    def __init__(self, df):
        self.df = df

    def dataset_summary(self):
        print(self.df.describe())
        print(self.df.isnull().sum())

    def binary_feature_survival_plots(self):

        binary_features = [
            "sex",
            "1st_class",
            "2nd_class",
            "3rd_class"
        ]

        for feature in binary_features:

            plt.figure(figsize=(6, 4))

            sns.barplot(
                x=feature,
                y='survived',
                data=self.df
            )

            plt.title(f'Survival Rate by {feature}')

            plt.show()

    def age_distribution(self):

        plt.figure(figsize=(8, 5))

        sns.histplot(
            self.df['age'],
            bins=30,
            kde=True
        )

        plt.title('Age Distribution')

        plt.show()

    def survival_by_age_group(self):

        viz_df = self.df.copy()

        viz_df['age_group'] = pd.cut(
            viz_df['age'],
            bins=np.arange(0, viz_df['age'].max() + 5, 5)
        )

        plt.figure(figsize=(12, 6))

        sns.countplot(
            data=viz_df,
            x='age_group',
            hue='survived'
        )

        plt.xticks(rotation=45)

        plt.title('Survival Count by Age Group (5-Year Bins)')

        plt.xlabel('Age Group')
        plt.ylabel('Count')

        plt.legend(title='Survived', labels=['No', 'Yes'])

        plt.show()

    def correlation_heatmap(self):

        plt.figure(figsize=(10, 6))

        sns.heatmap(
            self.df.select_dtypes(
                include=['int64', 'float64']
            ).corr(),
            annot=True,
            cmap='coolwarm'
        )

        plt.title('Correlation Heatmap')

        plt.show()

    def run_all(self):

        self.dataset_summary()
        self.binary_feature_survival_plots()
        self.age_distribution()
        self.survival_by_age_group()
        self.correlation_heatmap()