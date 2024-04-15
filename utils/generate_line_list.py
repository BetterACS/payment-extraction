
import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import KMeans

def generate_line_list(df:pd.DataFrame) -> list:
    y_coords = df[['y2', 'y1']].values
    silhouette_scores = []
    #บางทีปรับ range 2 ขึ้นทำให้ค่าเพี้ยน
    range_n_clusters = range(2, min(len(y_coords) - 1, 40))  # Define the range of cluster numbers to try
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(y_coords)
        
        silhouette_avg = silhouette_score(y_coords, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    
    #print(optimal_clusters)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=optimal_clusters) 
    df['cluster'] = kmeans.fit_predict(y_coords)

    # Sort dataframe by cluster and then by y1 to get the lines in order
    df.sort_values(by=["y1","y2"],inplace=True)
    
    line_dict = {}
    output_list = []
    
    # add values to line
    for index,x1,x2,y1,y2,text,cluster in df.values:
        line_dict[cluster] = line_dict.get(cluster,[]) + [(str(text),x1)]
    # sort line by x1
    for line in line_dict.values():
        line.sort(key=lambda x: x[1])
        output_list.append(" | ".join([x[0] for x in line]))
    return output_list