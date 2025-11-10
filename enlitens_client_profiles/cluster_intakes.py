"""
Cluster intake messages into distinct client segments.
Generate one persona per cluster for guaranteed diversity and authenticity.
"""
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from enlitens_client_profiles.config import ProfilePipelineConfig
from enlitens_client_profiles.data_ingestion import load_ingestion_bundle


def cluster_intakes(n_clusters: int = 100):
    """
    Cluster intake messages into n_clusters distinct segments.
    Returns cluster assignments and representative samples from each cluster.
    """
    
    project_root = Path("/home/antons-gs/enlitens-ai")
    config = ProfilePipelineConfig(project_root=project_root)
    
    print("="*80)
    print(f"CLUSTERING INTAKES INTO {n_clusters} DISTINCT CLIENT SEGMENTS")
    print("="*80 + "\n")
    
    # Load intakes
    print("Loading intake data...")
    bundle = load_ingestion_bundle(config)
    
    # Extract intake text from IntakeRecord objects
    intakes = []
    for intake_record in bundle.intakes:
        if intake_record.raw_text and len(intake_record.raw_text.strip()) > 20:
            intakes.append(intake_record.raw_text.strip())
    
    print(f"✅ Loaded {len(intakes)} intakes\n")
    
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    print("✅ Model loaded\n")
    
    # Generate embeddings
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(intakes, show_progress_bar=True, normalize_embeddings=True)
    print(f"✅ Generated {len(embeddings)} embeddings\n")
    
    # Cluster
    print(f"Clustering into {n_clusters} segments...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score (quality metric)
    silhouette = silhouette_score(embeddings, cluster_labels, sample_size=min(10000, len(embeddings)))
    print(f"✅ Clustering complete")
    print(f"   Silhouette score: {silhouette:.3f} (higher is better, range: -1 to 1)\n")
    
    # Organize clusters
    clusters = {}
    for i, (intake, label) in enumerate(zip(intakes, cluster_labels)):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            "index": i,
            "text": intake,
            "distance_to_center": float(np.linalg.norm(embeddings[i] - kmeans.cluster_centers_[label]))
        })
    
    # Sort each cluster by distance to center (closest = most representative)
    for label in clusters:
        clusters[label].sort(key=lambda x: x["distance_to_center"])
    
    # Print cluster summary
    print("="*80)
    print("CLUSTER SUMMARY")
    print("="*80 + "\n")
    
    cluster_sizes = [(label, len(members)) for label, members in clusters.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Cluster':>10} {'Size':>8} {'% of Total':>12} {'Representative Sample (first 100 chars)'}")
    print("-"*80)
    
    for label, size in cluster_sizes:
        pct = (size / len(intakes)) * 100
        sample = clusters[label][0]["text"][:100].replace("\n", " ")
        print(f"{label:>10} {size:>8} {pct:>11.1f}% {sample}")
    
    # Save cluster data
    output_dir = project_root / "enlitens_client_profiles" / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full cluster data
    cluster_data = {
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette),
        "total_intakes": len(intakes),
        "clusters": {}
    }
    
    for label, members in clusters.items():
        cluster_data["clusters"][str(label)] = {
            "size": len(members),
            "percentage": (len(members) / len(intakes)) * 100,
            "representative_samples": [
                {
                    "text": m["text"],
                    "distance": m["distance_to_center"]
                }
                for m in members[:5]  # Top 5 most representative
            ],
            "all_texts": [m["text"] for m in members]  # All texts for persona generation
        }
    
    output_file = output_dir / f"clusters_{n_clusters}.json"
    with open(output_file, "w") as f:
        json.dump(cluster_data, f, indent=2)
    
    print(f"\n✅ Saved cluster data to: {output_file}")
    
    # Print top 20 clusters with detailed samples
    print("\n" + "="*80)
    print("TOP 20 LARGEST CLUSTERS (Detailed View)")
    print("="*80 + "\n")
    
    for i, (label, size) in enumerate(cluster_sizes[:20], 1):
        pct = (size / len(intakes)) * 100
        print(f"\n{i}. CLUSTER #{label} ({size} intakes, {pct:.1f}%)")
        print("-"*80)
        
        # Show top 3 representative samples
        for j, member in enumerate(clusters[label][:3], 1):
            sample = member["text"][:200].replace("\n", " ")
            print(f"   Sample {j}: {sample}...")
        print()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Review the clusters above")
    print(f"2. Generate one persona per cluster (100 total)")
    print(f"3. Each persona will represent a real client segment")
    print(f"4. Guaranteed diversity (clusters are fundamentally different)")
    print(f"5. Guaranteed authenticity (using real intake language)\n")
    
    return cluster_data


if __name__ == "__main__":
    import sys
    
    # Allow command line argument for number of clusters
    n_clusters = 50  # Default to 50 for 224 intakes
    if len(sys.argv) > 1:
        try:
            n_clusters = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of clusters: {sys.argv[1]}, using default: 50")
    
    cluster_intakes(n_clusters=n_clusters)

