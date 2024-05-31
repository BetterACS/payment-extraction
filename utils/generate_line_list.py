import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


def generate_line_list(df: pd.DataFrame) -> list:
    df["center_x"] = df["x1"]+(df["x2"]-df["x1"]//2)
    df["center_y"] = df["y1"]+(df["y2"]-df["y1"]//2)
    y_coords = df[["center_y"]].values
    silhouette_scores = []
    #บางทีปรับ range 2 ขึ้นทำให้ค่าเพี้ยน
    range_n_clusters = range(len(y_coords)//2, min(len(y_coords) - 1, 40))  # Define the range of cluster numbers to try
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(y_coords)
        
        silhouette_avg = silhouette_score(y_coords, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    print(optimal_clusters)
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=optimal_clusters) 
    df['cluster'] = kmeans.fit_predict(y_coords)

    # Sort dataframe by cluster and then by y1 to get the lines in order
    df.sort_values(by=["y1"],inplace=True)
    
    line_dict = {}
    output_list = []

    # # add values to line
    for x1,x2,y1,y2,text,center_x,center_y,cluster, in df.values:
        line_dict[cluster] = line_dict.get(cluster,[]) + [(str(text),center_x)]
    # sort line by x1
    for line in line_dict.values():
        line.sort(key=lambda x: x[1])
        output_list.append(" | ".join([x[0] for x in line]))
    return output_list
