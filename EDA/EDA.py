import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)

def run_eda_df(df: pd.DataFrame, output_dir: str = "plots") -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nSample Rows:\n{df.head(1)}")

    #basic stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_df = df[numeric_cols].describe().T
    logger.info("Descriptive Statistics (Numeric Columns):\n%s", stats_df)

    # saved them to csv -- add more later
    stats_df.to_csv(os.path.join(output_dir, "numeric_descriptive_stats.csv"))

    #star rating distribution
    if 'stars_business' in df.columns:
        plt.figure(figsize=(8,6))
        sns.histplot(df['stars_business'].dropna(), bins=20, kde=True)
        plt.title("Distribution of Restaurant Business Star Ratings")
        plt.xlabel("Star Rating")
        plt.ylabel("Frequency")
        out_path = os.path.join(output_dir, "dist_stars_business.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot: {out_path}")
    else:
        logger.warning("No 'stars_business' column found for star rating distribution.")

    # 3) Box plot of star ratings by region
    if 'Region' in df.columns and 'stars_business' in df.columns:
        plt.figure(figsize=(10,6))
        sns.boxplot(data=df, x='Region', y='stars_business', showfliers=False)
        plt.title("Box Plot of Star Ratings by Region")
        plt.xlabel("Region")
        plt.ylabel("Star Rating")
        out_path = os.path.join(output_dir, "box_stars_by_region.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()

        # violin plot:
        plt.figure(figsize=(10,6))
        sns.violinplot(data=df, x='Region', y='stars_business', inner='quartile', cut=0)
        plt.title("Violin Plot: Star Ratings by Region")
        plt.xlabel("Region")
        plt.ylabel("Star Rating")
        out_path2 = os.path.join(output_dir, "violin_stars_by_region.png")
        plt.savefig(out_path2, dpi=100, bbox_inches="tight")
        plt.close()

        logger.info("Saved region-based box & violin plots.")
    else:
        logger.info("Skipping region-based rating plots (missing 'Region' or 'stars_business').")

    # avg stars by region
    if 'Region' in df.columns and 'stars_business' in df.columns:
        region_agg = df.groupby('Region')['stars_business'].mean().sort_values(ascending=False)
        logger.info("\nAverage Star Rating by Region:\n%s", region_agg)
        # Plot
        plt.figure(figsize=(8,4))
        region_agg.plot(kind='bar', color='skyblue')
        plt.title("Average Star Rating by Region")
        plt.xlabel("Region")
        plt.ylabel("Mean Star Rating")
        out_path = os.path.join(output_dir, "mean_stars_by_region.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved bar chart of average stars by region: {out_path}")

    if 'SentiScore' in df.columns and 'stars_business' in df.columns:
        # Group by SentiScore and see the average star rating
        senti_agg = df.groupby('SentiScore')['stars_business'].mean().sort_values(ascending=False)
        logger.info("\nAverage Star Rating by SentiScore:\n%s", senti_agg)

        plt.figure(figsize=(6,4))
        senti_agg.plot(kind='bar', color='orange')
        plt.title("Average Star Rating by SentiScore")
        plt.xlabel("SentiScore")
        plt.ylabel("Mean Star Rating")
        out_path = os.path.join(output_dir, "mean_stars_by_senti.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved bar chart of average stars by SentiScore: {out_path}")

    # correlation heatmap
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=False, cmap="YlGnBu", square=True)
        plt.title("Correlation Heatmap of Numeric Features")
        out_path = os.path.join(output_dir, "numeric_corr_heatmap.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved numeric correlation heatmap: {out_path}")
    else:
        logger.info("Not enough numeric columns for correlation heatmap.")

    # Scatter of review count vs. star rating
    if 'review_count' in df.columns and 'stars_business' in df.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x='review_count', y='stars_business', alpha=0.3)
        plt.title("Review Count vs. Star Rating")
        plt.xlabel("Review Count")
        plt.ylabel("Star Rating")
        out_path = os.path.join(output_dir, "scatter_reviewcount_stars.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved scatter plot of review_count vs. stars_business: {out_path}")

    # snetiment score plot
    if 'SentiScore' in df.columns:
        plt.figure(figsize=(6,4))
        df['SentiScore'].value_counts().plot(kind='bar', color='green')
        plt.title("Distribution of SentiScore")
        plt.xlabel("SentiScore Category")
        plt.ylabel("Count")
        out_path = os.path.join(output_dir, "count_senti_score.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved count plot of SentiScore: {out_path}")

    logger.info("Advanced EDA complete. Plots and stats saved to '%s'.", output_dir)



def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    )
    run_eda_df()

if __name__ == "__main__":
    main()