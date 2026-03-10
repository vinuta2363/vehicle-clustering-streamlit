import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from model import run_model

st.title("Vehicle Type Clustering using CNN Autoencoder")

st.write("Unsupervised Vehicle Clustering System")

dataset_path = st.text_input("Enter Dataset Folder Path")

if st.button("Run Model"):

    sil, db, ch, images, labels = run_model(dataset_path)

    st.subheader("Clustering Metrics")

    st.write("Silhouette Score:", sil)
    st.write("Davies Bouldin Index:", db)
    st.write("Calinski Harabasz Score:", ch)

    st.subheader("Cluster Visualization")

    clusters = np.unique(labels)

    for c in clusters:

        st.write(f"Cluster {c}")

        idx = np.where(labels == c)[0]

        fig = plt.figure(figsize=(6,6))

        for i in range(min(9,len(idx))):

            plt.subplot(3,3,i+1)
            plt.imshow(images[idx[i]])
            plt.axis("off")

        st.pyplot(fig)

