import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from skimage import io
from sklearn.preprocessing import StandardScaler

def pca_band_selection_batch(X, num_selected_bands, batch_size=1000):
    """
    Selects the most important bands using PCA with batch processing.

    Parameters:
        X: np.ndarray
            Input hyperspectral data of shape (num_samples, num_bands).
        num_selected_bands: int
            Number of bands to select.
        batch_size: int
            Number of samples per batch.

    Returns:
        selected_band_indices: np.ndarray
            Indices of the most important bands across all batches.
    """
    num_batches = int(np.ceil(X.shape[0] / batch_size))
    total_contribution = np.zeros(X.shape[1])

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, X.shape[0])
        batch = X[batch_start:batch_end]

        # PCA for the current batch
        pca = PCA(n_components=batch.shape[1])
        pca.fit(batch)

        # Sum contributions of each band across all batches
        contribution = np.abs(pca.components_)
        total_contribution += np.sum(contribution, axis=0)

    # Rank bands by total contribution
    band_indices = np.argsort(-total_contribution)
    selected_band_indices = band_indices[:num_selected_bands]

    return selected_band_indices

def loadData(list_files, type='psp'):
    num_ch = 8
    chs = [3,4,5,6,7,8,9,10]
    # chs = [15,16,17,18,19,20,21,22]
    img_size = 128
    
    total_data = []
    for file in list_files:
        set_temp = np.zeros((img_size, img_size, num_ch), dtype=np.uint8)
        img_ = io.imread(f'./data/20230525/{file}.tif')
        x = np.moveaxis(img_, 0, 2)
        for idx in range(num_ch):
            set_temp[:,:,idx] = x[:,:,chs[idx]]
        total_data.append(set_temp)

    total_data = np.array(total_data)
    return total_data

# Example usage
if __name__ == "__main__":
    path_to_csv = './data/labels_20230525.csv'
    data = pd.read_csv(path_to_csv)
    list_files = data['patient_id'].values
    data = loadData(list_files)

    H, W, Z, num_bands = data.shape
    
    data_flat = data.reshape(-1, num_bands)  # Flatten spatial dimensions, shape: (H*W, Bands)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data_flat)    
    
    # Example: 6000개의 샘플을 배치 크기 1000으로 나눠 처리
    selected_bands = pca_band_selection_batch(X_normalized, num_selected_bands=7, batch_size=2000)
    # selected_bands = pca_band_selection_batch(data_flat, num_selected_bands=7, batch_size=2000)
    print("Selected Band Indices:", selected_bands)
    ###### sp
    # [3, 2, 4, 5, 6, 0, 1] < scale (work fine)
    # [2 1 5 3 6 0 4 7] < not scale (test next)
    ###### psp
    # [3, 4, 7, 0, 2, 6, 5] < scale
    # [1 0 2 5 3 4 7 6] < not scale (test next)
    # 4 랑 1의 교환? 전체 교환 필요