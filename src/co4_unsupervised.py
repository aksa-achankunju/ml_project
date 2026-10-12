"""
RoadSafe AI - CO4 Unsupervised Learning & Anomaly Detection
Implements:
1. Spatial & Risk Clustering:
   - K-Means Clustering (Elbow method inertia & Silhouette score analysis)
   - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
   - Hierarchical / Agglomerative Clustering
2. Dimensionality Reduction:
   - PCA (Principal Component Analysis - Scree Plot & Explained Variance)
   - t-SNE (t-Distributed Stochastic Neighbor Embedding - Non-linear 2D projection)
3. Anomaly Detection:
   - Isolation Forest (High-impact anomaly / outlier accident hazard identification)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

class CO4UnsupervisedExperimenter:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca_model = None
        self.isolation_forest_model = None
        self.results = {}

    def run_kmeans_analysis(self, df_sample, k_range=range(3, 9)):
        """
        Runs K-Means clustering on geographic and risk features,
        evaluates inertia (Elbow method) and Silhouette scores.
        """
        coords = df_sample[['Start_Lat', 'Start_Lng']].values
        
        elbow_data = []
        best_k = 5
        best_silhouette = -1

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(coords)
            inertia = float(km.inertia_)
            
            # Compute silhouette on a subset for speed
            sample_idx = np.random.RandomState(42).choice(len(coords), size=min(2000, len(coords)), replace=False)
            sil = float(silhouette_score(coords[sample_idx], km.labels_[sample_idx]))
            
            elbow_data.append({
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil, 4)
            })

        # Fit chosen production K-Means (k=5)
        self.kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = self.kmeans_model.fit_predict(coords)
        
        cluster_summaries = []
        for c in range(5):
            mask = (labels == c)
            cluster_summaries.append({
                "cluster_id": c,
                "centroid_lat": round(float(self.kmeans_model.cluster_centers_[c][0]), 4),
                "centroid_lng": round(float(self.kmeans_model.cluster_centers_[c][1]), 4),
                "sample_count": int(np.sum(mask)),
                "avg_severity": round(float(df_sample.loc[mask, 'Severity'].mean()), 2),
                "avg_temperature": round(float(df_sample.loc[mask, 'Temperature(F)'].mean()), 1)
            })

        self.results['kmeans'] = {
            "elbow_curve": elbow_data,
            "chosen_k": 5,
            "clusters": cluster_summaries
        }
        return self.results['kmeans']

    def run_dbscan_analysis(self, df_sample, eps=0.7, min_samples=25):
        """
        Density-Based Clustering on spatial coordinates to identify dense accident corridors
        and isolate non-cluster noise points.
        """
        # Run on a 5,000-point sample for fast spatial density profiling
        sub_sample = df_sample.sample(n=min(5000, len(df_sample)), random_state=42).copy()
        coords = sub_sample[['Start_Lat', 'Start_Lng']].values

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = dbscan.fit_predict(coords)
        self.dbscan_model = dbscan

        unique_labels = set(db_labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = int(np.sum(db_labels == -1))

        # Summarize dense corridors
        corridors = []
        for lab in unique_labels:
            if lab == -1:
                continue
            mask = (db_labels == lab)
            corridors.append({
                "corridor_id": int(lab),
                "incident_density": int(np.sum(mask)),
                "center_lat": round(float(np.mean(coords[mask, 0])), 4),
                "center_lng": round(float(np.mean(coords[mask, 1])), 4),
                "avg_severity": round(float(sub_sample.iloc[mask]['Severity'].mean()), 2)
            })

        # Sort by incident density
        corridors = sorted(corridors, key=lambda x: x['incident_density'], reverse=True)[:8]

        self.results['dbscan'] = {
            "total_corridors_found": n_clusters,
            "noise_incidents_isolated": n_noise,
            "noise_ratio": round(n_noise / len(coords), 3),
            "top_dense_corridors": corridors
        }
        return self.results['dbscan']

    def run_pca_and_tsne(self, X_train, n_components=10):
        """
        Runs PCA for scree plot explained variance and t-SNE for 2D latent manifold discovery.
        """
        X_proc = self.preprocessor.transform(X_train)

        # PCA
        pca = PCA(n_components=min(n_components, X_proc.shape[1]), random_state=42)
        pca.fit(X_proc)
        self.pca_model = pca

        exp_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        pca_summary = [
            {
                "component": f"PC{i+1}",
                "explained_variance_ratio": round(float(exp_var[i]), 4),
                "cumulative_variance_ratio": round(float(cum_var[i]), 4)
            }
            for i in range(len(exp_var))
        ]

        # t-SNE on a small sample of 1,000 points
        sample_indices = np.random.RandomState(42).choice(len(X_proc), size=min(1000, len(X_proc)), replace=False)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=800)
        X_tsne_2d = tsne.fit_transform(X_proc[sample_indices])

        self.results['dimensionality_reduction'] = {
            "pca_explained_variance": pca_summary,
            "total_variance_explained_top10": round(float(cum_var[-1]), 4),
            "tsne_sample_size": len(sample_indices),
            "tsne_embedding_bounds": {
                "x_min": round(float(np.min(X_tsne_2d[:, 0])), 2),
                "x_max": round(float(np.max(X_tsne_2d[:, 0])), 2),
                "y_min": round(float(np.min(X_tsne_2d[:, 1])), 2),
                "y_max": round(float(np.max(X_tsne_2d[:, 1])), 2)
            }
        }
        return self.results['dimensionality_reduction']

    def run_anomaly_detection(self, X_train, contamination=0.03):
        """
        Trains Isolation Forest to detect extreme/anomalous hazard conditions in accident profiles.
        """
        X_proc = self.preprocessor.transform(X_train)
        iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
        iso.fit(X_proc)
        self.isolation_forest_model = iso

        preds = iso.predict(X_proc)
        scores = iso.decision_function(X_proc)

        num_anomalies = int(np.sum(preds == -1))
        self.results['anomaly_detection'] = {
            "model": "Isolation Forest",
            "contamination_rate": contamination,
            "anomalies_detected": num_anomalies,
            "normal_instances": int(np.sum(preds == 1)),
            "anomaly_score_mean": round(float(np.mean(scores)), 4),
            "anomaly_score_std": round(float(np.std(scores)), 4),
            "significance": "Flags extreme risk outliers (multi-vehicle pileups, blizzard hazards) where standard traffic rules break down."
        }
        return self.results['anomaly_detection']
