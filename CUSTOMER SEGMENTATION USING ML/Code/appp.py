from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np
import os
import io
import base64

app = Flask(__name__)

# Helper function to generate plots as base64 strings
def plot_to_img_tag(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['dataset']
    if not file:
        return "No file uploaded", 400

    data = pd.read_excel(file)

    # Data Cleaning
    data.dropna(subset=['CustomerID'], inplace=True)
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
    data['CustomerID'] = data['CustomerID'].astype(int)
    data['Total'] = data['Quantity'] * data['UnitPrice']

    # RFM Calculation
    snapshot_date = data['InvoiceDate'].max() + pd.DateOffset(days=1)
    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Total': 'sum'
    })
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'Total': 'MonetaryValue'
    }, inplace=True)

    # Log Transformation
    rfm_log = rfm[['Recency', 'Frequency', 'MonetaryValue']].apply(lambda x: np.log1p(x))

    # KMeans Clustering with Elbow Method
    inertia = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_log)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(range(2, 11), inertia, marker='o')
    ax.set_title('Elbow Curve')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    elbow_plot = plot_to_img_tag(fig)
    plt.close(fig)

    # Final Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_log)
    rfm_log['Cluster'] = rfm['Cluster']
    
    # Get cluster centroids and find nearest customer to each centroid
    centroids = kmeans.cluster_centers_
    
    # Find representative customer (nearest to centroid) for each cluster
    representative_customers = []
    for i in range(kmeans.n_clusters):
        # Calculate distances from centroid
        distances = np.linalg.norm(rfm_log[rfm_log['Cluster'] == i][['Recency', 'Frequency', 'MonetaryValue']] - centroids[i], axis=1)
        # Get index of nearest customer
        nearest_idx = distances.argmin()
        # Get the actual CustomerID
        cluster_customers = rfm_log[rfm_log['Cluster'] == i]
        representative_id = cluster_customers.iloc[nearest_idx].name
        representative_customers.append(representative_id)
    
    # Create centroids DataFrame with representative CustomerIDs
    centroids_df = pd.DataFrame(centroids, columns=['Recency', 'Frequency', 'MonetaryValue'])
    centroids_df['Cluster'] = centroids_df.index
    centroids_df['CustomerID'] = [f"Centroid {i} (Rep: {rep})" for i, rep in enumerate(representative_customers)]

    # Cluster Names based on RFM scores
    cluster_names = {
        0: "Champion Customers",
        1: "Potential Loyalists",
        2: "At-Risk Customers",
        3: "Lost Customers"
    }

    # Interactive 3D Scatter plot using Plotly
    rfm_plot = rfm_log.copy()
    rfm_plot['CustomerID'] = rfm.index

    fig3d = px.scatter_3d(
        rfm_plot,
        x='Recency', y='Frequency', z='MonetaryValue',
        color='Cluster',
        hover_data=['CustomerID', 'Recency', 'Frequency', 'MonetaryValue'],
        title='Interactive 3D Cluster Plot with Centroids',
        width=900, height=700,
        color_discrete_map=cluster_names
    )
    
    # Add centroids to the plot with enhanced hover information
    fig3d.add_scatter3d(
        x=centroids_df['Recency'],
        y=centroids_df['Frequency'],
        z=centroids_df['MonetaryValue'],
        mode='markers',
        marker=dict(
            size=12,
            color='black',
            symbol='diamond',
            line=dict(width=2, color='white')
        ),
        name='Cluster Centers',
        hovertext=centroids_df['CustomerID'],
        customdata=centroids_df[['Recency', 'Frequency', 'MonetaryValue']],
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>" +
            "Recency: %{customdata[0]:.2f}<br>" +
            "Frequency: %{customdata[1]:.2f}<br>" +
            "MonetaryValue: %{customdata[2]:.2f}<br>" +
            "<extra></extra>"
        )
    )
    
    fig3d.update_layout(
        legend_title_text='Customer Segments',
        scene=dict(
            xaxis_title='Recency (log)',
            yaxis_title='Frequency (log)',
            zaxis_title='MonetaryValue (log)'
        )
    )
    plotly_3d_html = fig3d.to_html(full_html=False)

    # Prepare cluster analysis summary with representative customers
    cluster_summary = rfm_log.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean'
    }).reset_index().rename(columns={
        'Recency': 'AvgRecency',
        'Frequency': 'AvgFrequency',
        'MonetaryValue': 'AvgMonetaryValue'
    })
    cluster_summary['Customers'] = rfm['Cluster'].value_counts().sort_index().values
    cluster_summary['ClusterName'] = cluster_summary['Cluster'].map(cluster_names)
    cluster_summary['RepresentativeCustomer'] = representative_customers

    # Convert log-scaled RFM table to HTML
    rfm_log['CustomerID'] = rfm.index
    rfm_log_table = rfm_log[['CustomerID', 'Recency', 'Frequency', 'MonetaryValue', 'Cluster']].reset_index(drop=True)
    rfm_table_html = rfm_log_table.to_html(classes='table table-striped', index=False)

    # Create Excel download
    excel_buffer = io.BytesIO()
    rfm_log_table.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    excel_b64 = base64.b64encode(excel_buffer.read()).decode('utf-8')

    return render_template(
        'result.html',
        elbow_plot=elbow_plot,
        cluster_plot=None,
        cluster_3d_html=plotly_3d_html,
        table=rfm_table_html,
        cluster_summary=cluster_summary.to_dict(orient='records'),
        cluster_names=cluster_names,
        excel_data=excel_b64
    )

if __name__ == '__main__':
    app.run(debug=True)