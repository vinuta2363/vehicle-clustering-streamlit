import os
import random
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.layers import UpSampling2D, Reshape, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


def run_model(dataset_path):

    IMG_SIZE = 32
    MAX_IMAGES = 1000
    CLUSTERS = 6
    EPOCHS = 40
    BATCH_SIZE = 64
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    images = []

    files = [f for f in os.listdir(dataset_path)
             if f.lower().endswith((".jpg",".png",".jpeg"))][:MAX_IMAGES]

    for f in files:

        img = cv2.imread(os.path.join(dataset_path, f))

        if img is None:
            continue

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)

    images = np.array(images,dtype="float32") / 255.0

    # =============================
    # AUTOENCODER
    # =============================

    input_img = Input(shape=(IMG_SIZE,IMG_SIZE,3))

    x = Conv2D(32,3,padding="same")(input_img)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64,3,padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(128,3,padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(2)(x)

    shape_before_flatten = x.shape

    x = Flatten()(x)

    encoded = Dense(
        96,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        activity_regularizer=regularizers.l1(2e-4)
    )(x)

    decoder_input = Dense(
        shape_before_flatten[1]*
        shape_before_flatten[2]*
        shape_before_flatten[3],
        activation="relu"
    )(encoded)

    decoded = Reshape((
        shape_before_flatten[1],
        shape_before_flatten[2],
        shape_before_flatten[3]
    ))(decoder_input)

    decoded = UpSampling2D(2)(decoded)
    decoded = Conv2D(128,3,padding="same")(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation("relu")(decoded)

    decoded = UpSampling2D(2)(decoded)
    decoded = Conv2D(64,3,padding="same")(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation("relu")(decoded)

    decoded = UpSampling2D(2)(decoded)
    decoded = Conv2D(32,3,padding="same")(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation("relu")(decoded)

    decoded = Conv2D(3,3,activation="sigmoid",padding="same")(decoded)

    autoencoder = Model(input_img,decoded)
    encoder = Model(input_img,encoded)

    autoencoder.compile(
        optimizer=Adam(1e-4),
        loss="mse"
    )

    autoencoder.fit(
        images,
        images,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=0
    )

    # =============================
    # FEATURE EXTRACTION
    # =============================

    features = encoder.predict(images)

    features = normalize(features)

    # =============================
    # PCA
    # =============================

    pca = PCA(n_components=0.95, whiten=True, random_state=SEED)
    features_pca = pca.fit_transform(features)

    # =============================
    # OUTLIER DETECTION
    # =============================

    iso = IsolationForest(
        contamination=0.05,
        random_state=SEED
    )

    outliers = iso.fit_predict(features_pca)

    mask = outliers == 1

    features_clean = features_pca[mask]
    images_clean = images[mask]

    # =============================
    # KMEANS
    # =============================

    kmeans = KMeans(
        n_clusters=CLUSTERS,
        init="k-means++",
        n_init=100,
        max_iter=1000,
        random_state=SEED
    )

    labels = kmeans.fit_predict(features_clean)

    # =============================
    # METRICS
    # =============================

    sil = silhouette_score(features_clean,labels)

    db = davies_bouldin_score(features_clean,labels)

    ch = calinski_harabasz_score(features_clean,labels)

    return sil, db, ch, images_clean, labels
