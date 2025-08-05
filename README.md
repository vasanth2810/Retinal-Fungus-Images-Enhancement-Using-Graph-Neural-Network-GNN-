# Install required packages
import subprocess
import sys

def install_packages():
    required_packages = [
        'torch',
        'torchvision',
        'torchaudio',
        'scikit-image',
        'pandas',
        'seaborn',
        'matplotlib',
        'opencv-python'
    ]

    # First install base packages
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Install PyTorch Geometric with dependencies
    try:
        from torch_geometric.data import Data
    except ImportError:
        import torch
        TORCH_VERSION = torch.__version__.split('+')[0]
        CUDA = 'cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'

        # For PyTorch 2.6.0 with CUDA 12.4
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                             "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv",
                             "-f", f"https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA}.html"])

        # Install main package
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])

install_packages()
# Install required packages
import subprocess
import sys

def install_packages():
    required_packages = [
        'torch',
        'torchvision',
        'torchaudio',
        'scikit-image',
        'pandas',
        'seaborn',
        'matplotlib',
        'opencv-python-headless',
        'opencv-contrib-python-headless',
        'scikit-learn'
    ]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

# Main Imports
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.pool import knn_graph
from skimage import feature, filters, exposure, color
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import traceback
import warnings
from scipy import ndimage
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata

# Suppress warnings
warnings.filterwarnings("ignore")

class PerformanceMetrics:
    @staticmethod
    def compute_psnr(original, enhanced):
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
        return psnr(original, enhanced, data_range=1.0)

    @staticmethod
    def compute_ssim(original, enhanced):
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
        return ssim(original, enhanced, data_range=1.0, channel_axis=2 if original.ndim == 3 else None)

    @staticmethod
    def compute_mae(original, enhanced):
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
        return np.mean(np.abs(original - enhanced))

    @staticmethod
    def compute_nrmse(original, enhanced):
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
        mse = mean_squared_error(original, enhanced)
        return np.sqrt(mse) / (np.max(original) - np.min(original))

    @staticmethod
    def compute_all_metrics(original, processed):
        return {
            'PSNR': PerformanceMetrics.compute_psnr(original, processed),
            'SSIM': PerformanceMetrics.compute_ssim(original, processed),
            'MAE': PerformanceMetrics.compute_mae(original, processed),
            'NRMSE': PerformanceMetrics.compute_nrmse(original, processed)
        }

class Config:
    def __init__(self):
        self.input_dir = '/content/drive/MyDrive/gan/train'
        self.output_dir = '/content/drive/MyDrive/improved_gnn_modal_finalized'

        self.subdirs = {
            'grayscale': 'grayscale',
            'enhanced': 'enhanced',
            'vessel': 'vessel_enhanced',
            'gnn': 'gnn_enhanced',
            'color': 'color_restored',
            'final': 'final_output',
            'comparisons': 'comparisons',
            'metrics': 'performance_metrics'
        }

        # Model parameters
        self.num_node_features = 32
        self.gnn_hidden_channels = 128
        self.gnn_output_channels = 64
        self.patch_size = 9
        self.graph_step = 4
        self.knn_k = 8

        # Training parameters
        self.batch_size = 2
        self.epochs = 30
        self.learning_rate = 0.0001
        self.dropout_rate = 0.2

        # Image processing
        self.frangi_sigmas = (1, 2, 3)
        self.clip_limit = 0.03

        self._create_directories()
        self.metrics_df = pd.DataFrame(columns=[
            'Image', 'Original_PSNR', 'Original_SSIM', 'Original_MAE', 'Original_NRMSE',
            'Enhanced_PSNR', 'Enhanced_SSIM', 'Enhanced_MAE', 'Enhanced_NRMSE',
            'GNN_PSNR', 'GNN_SSIM', 'GNN_MAE', 'GNN_NRMSE',
            'Final_PSNR', 'Final_SSIM', 'Final_MAE', 'Final_NRMSE'
        ])

    def _create_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for name, subdir in self.subdirs.items():
            path = os.path.join(self.output_dir, subdir)
            os.makedirs(path, exist_ok=True)

class ImageProcessor:
    @staticmethod
    def to_grayscale(img):
        return color.rgb2gray(img)

    @staticmethod
    def enhance_contrast(img, clip_limit=0.03):
        return exposure.equalize_adapthist(img, clip_limit=clip_limit)

    @staticmethod
    def extract_vessels(img, sigmas=(1, 2, 3)):
        vessel_maps = []
        weights = [0.3, 0.4, 0.3]
        for sigma, weight in zip(sigmas, weights):
            vessel_map = filters.frangi(img, sigmas=[sigma], black_ridges=False)
            vessel_maps.append(vessel_map * weight)
        return np.sum(vessel_maps, axis=0)

    @staticmethod
    def restore_color(grayscale_img, original_img):
        lab = color.rgb2lab(original_img)
        L = grayscale_img * 100
        a = cv2.bilateralFilter(lab[:, :, 1].astype(np.float32), d=9, sigmaColor=0.2, sigmaSpace=9)
        b = cv2.bilateralFilter(lab[:, :, 2].astype(np.float32), d=9, sigmaColor=0.2, sigmaSpace=9)
        return np.clip(color.lab2rgb(np.stack([L, a, b], axis=2)), 0, 1)

class EnhancedGNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = GCNConv(config.num_node_features, config.gnn_hidden_channels)
        self.conv2 = GCNConv(config.gnn_hidden_channels, config.gnn_hidden_channels)
        self.conv3 = GCNConv(config.gnn_hidden_channels, config.gnn_hidden_channels)

        self.attention = nn.Sequential(
            nn.Linear(config.gnn_hidden_channels, config.gnn_hidden_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(config.gnn_hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(config.gnn_hidden_channels, config.gnn_hidden_channels // 2)
        self.fc2 = nn.Linear(config.gnn_hidden_channels // 2, config.gnn_output_channels)
        self.fc3 = nn.Linear(config.gnn_output_channels, 1)

        self.bn1 = nn.BatchNorm1d(config.gnn_hidden_channels)
        self.bn2 = nn.BatchNorm1d(config.gnn_hidden_channels)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.2)
        x1 = self.bn1(x1)
        x1 = self.dropout(x1)

        x2 = F.leaky_relu(self.conv2(x1, edge_index), negative_slope=0.2)
        x2 = self.bn2(x2 + x1)
        x2 = self.dropout(x2)

        x3 = F.leaky_relu(self.conv3(x2, edge_index), negative_slope=0.2)
        attn = self.attention(x3)
        x3 = x3 * attn

        x_pool = global_mean_pool(x3, batch)

        x = F.leaky_relu(self.fc1(x_pool), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

class RetinalGraphDataset(Dataset):
    def __init__(self, img_dir, config, num_samples=None):
        self.config = config
        self.image_files = self._get_image_files(img_dir, num_samples)
        self.scaler = MinMaxScaler()

    def _get_image_files(self, img_dir, num_samples):
        image_files = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, f)
                    img = cv2.imread(full_path)
                    if img is not None:
                        image_files.append(full_path)
                        if num_samples and len(image_files) >= num_samples:
                            break
        return image_files[:num_samples] if num_samples else image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        grayscale = ImageProcessor.to_grayscale(img)
        enhanced = ImageProcessor.enhance_contrast(grayscale, self.config.clip_limit)
        vessels = ImageProcessor.extract_vessels(enhanced, self.config.frangi_sigmas)
        return self._create_graph(enhanced, vessels, grayscale)

    def _create_graph(self, enhanced_img, vessel_img, grayscale_img):
        h, w = enhanced_img.shape
        step = self.config.graph_step
        patch_size = self.config.patch_size

        features, positions = [], []
        for i in range(0, h, step):
            for j in range(0, w, step):
                patch = enhanced_img[
                    max(0, i-patch_size//2):min(h, i+patch_size//2+1),
                    max(0, j-patch_size//2):min(w, j+patch_size//2+1)
                ]
                patch_uint8 = (patch * 255).astype(np.uint8)

                # Base features
                feat = [
                    enhanced_img[i, j], vessel_img[i, j], grayscale_img[i, j],
                    np.mean(patch), np.std(patch), np.max(patch)-np.min(patch),
                    filters.sobel(patch).mean(), i/h, j/w,
                    *np.histogram(patch, bins=5, range=(0, 1))[0],
                    *feature.local_binary_pattern(patch, 8, 1).flatten()[:15]
                ]

                # Texture features using graycomatrix
                try:
                    from skimage.feature import graycomatrix, graycoprops
                    glcm = graycomatrix(patch_uint8, [1], [0], symmetric=True, normed=True)
                    contrast = graycoprops(glcm, 'contrast')[0, 0]
                    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                    feat.extend([contrast, dissimilarity, homogeneity])
                except:
                    # Fallback features
                    feat.extend([patch.mean(), patch.std(), patch.var()])

                feat = feat[:self.config.num_node_features]
                if len(feat) < self.config.num_node_features:
                    feat += [0]*(self.config.num_node_features - len(feat))
                features.append(feat)
                positions.append([i, j])

        features = self.scaler.fit_transform(np.array(features))
        edge_index = knn_graph(
            torch.tensor(np.array(positions), dtype=torch.float),
            k=self.config.knn_k,
            loop=True
        )

        return Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
            original_shape=(h, w),
            pos=torch.tensor(positions, dtype=torch.float32)
        )

class EnhancedRetinalPipeline:
    def __init__(self, config):
        self.config = config
        self.processor = ImageProcessor()
        self.metrics = PerformanceMetrics()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = RetinalGraphDataset(config.input_dir, config)
        if len(self.dataset) == 0:
            raise ValueError("No images found in directory")

    def train_gnn(self):
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: Batch.from_data_list(batch)
        )

        model = EnhancedGNNModel(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(self.device))

        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in tqdm(loader, desc=f"Epoch {epoch}/{self.config.epochs}"):
                batch = batch.to(self.device)
                optimizer.zero_grad()

                out = model(batch)
                target = global_mean_pool(batch.x[:, 1].unsqueeze(1), batch.batch)

                loss = criterion(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.config.output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Best Loss: {best_loss:.4f}')

        model.load_state_dict(torch.load(os.path.join(self.config.output_dir, 'best_model.pth')))
        return model

    def process_image(self, img_path, model):
        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            filename = os.path.basename(img_path)

            grayscale = self.processor.to_grayscale(img)
            enhanced = self.processor.enhance_contrast(grayscale, self.config.clip_limit)
            vessels = self.processor.extract_vessels(enhanced, self.config.frangi_sigmas)

            graph = self.dataset._create_graph(enhanced, vessels, grayscale)
            h, w = graph.original_shape

            graph = graph.to(self.device)
            with torch.no_grad():
                model.eval()
                output = model(graph)

            # Get nodes and output values
            nodes = graph.pos.cpu().numpy()
            output_values = output.cpu().numpy().reshape(-1)

            # Ensure matching dimensions
            if len(nodes) != len(output_values):
                min_len = min(len(nodes), len(output_values))
                nodes = nodes[:min_len]
                output_values = output_values[:min_len]

            # Create grid for interpolation
            grid_x, grid_y = np.mgrid[0:h, 0:w]

            # Use nearest neighbor interpolation for stability
            gnn_output = griddata(
                nodes[:, :2],
                output_values,
                (grid_x, grid_y),
                method='nearest',  # Changed to nearest neighbor for stability
                fill_value=0
            )

            vessel_mask = (vessels > 0.2).astype(float)
            gnn_enhanced = enhanced * (1 - vessel_mask) + (enhanced + gnn_output * 0.7) * vessel_mask
            gnn_enhanced = np.clip(gnn_enhanced, 0, 1)

            final = self.processor.restore_color(gnn_enhanced, img)

            self._save_outputs(filename, grayscale, enhanced, vessels, gnn_enhanced, final)
            self._save_comparison(filename, img, grayscale, enhanced, vessels, gnn_enhanced, final)

            metrics = {
                'Image': filename,
                **PerformanceMetrics.compute_all_metrics(img, img),
                **{'Enhanced_'+k: v for k,v in PerformanceMetrics.compute_all_metrics(img, self.processor.restore_color(enhanced, img)).items()},
                **{'GNN_'+k: v for k,v in PerformanceMetrics.compute_all_metrics(img, self.processor.restore_color(gnn_enhanced, img)).items()},
                **{'Final_'+k: v for k,v in PerformanceMetrics.compute_all_metrics(img, final).items()}
            }
            self.config.metrics_df.loc[len(self.config.metrics_df)] = metrics

            return final

        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {str(e)}")
            traceback.print_exc()
            return None

    def _save_outputs(self, filename, *images):
        names = ['grayscale', 'enhanced', 'vessel', 'gnn', 'final']
        for name, img in zip(names, images):
            output_path = os.path.join(self.config.output_dir, self.config.subdirs[name], filename)
            if img.ndim == 2:
                cv2.imwrite(output_path, (img * 255).astype(np.uint8))
            else:
                cv2.imwrite(output_path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    def _save_comparison(self, filename, original, *processed_images):
        enhanced_color = self.processor.restore_color(processed_images[1], original)
        gnn_enhanced_color = self.processor.restore_color(processed_images[3], original)

        titles = ['Original', 'Grayscale', 'Enhanced', 'Vessels', 'GNN Enhanced', 'Final',
                 'Enhanced Color', 'GNN Enhanced Color']
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()

        for ax, img, title in zip(axes,
                                [original, processed_images[0], processed_images[1],
                                processed_images[2], processed_images[3], processed_images[4],
                                enhanced_color, gnn_enhanced_color],
                                titles):
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, self.config.subdirs['comparisons'], filename))
        plt.close()

    def _plot_metrics(self):
        df = self.config.metrics_df
        metrics = ['PSNR', 'SSIM', 'MAE', 'NRMSE']

        # Create individual plots for each metric
        for metric in metrics:
            plt.figure(figsize=(10, 6))

            # Get values for Enhanced and GNN methods
            enhanced_vals = df[f'Enhanced_{metric}']
            gnn_vals = df[f'GNN_{metric}']

            # Create bar positions
            x = np.arange(len(df))
            width = 0.35

            # Plot bars
            plt.bar(x - width/2, enhanced_vals, width, label='Enhanced', alpha=0.7)
            plt.bar(x + width/2, gnn_vals, width, label='GNN Enhanced', alpha=0.7)

            # Customize plot
            plt.xlabel('Image Index')
            plt.ylabel(metric)
            plt.title(f'{metric} Comparison: Enhanced vs GNN Enhanced')
            plt.xticks(x, df['Image'], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save and show
            plt.savefig(os.path.join(self.config.output_dir, f'{metric.lower()}_comparison.png'))
            plt.show()

        # Also save the combined metrics plot
        self._plot_combined_metrics()

    def _plot_combined_metrics(self):
        df = self.config.metrics_df
        metrics = ['PSNR', 'SSIM', 'MAE', 'NRMSE']

        plt.figure(figsize=(12, 8))
        x = np.arange(len(metrics))
        width = 0.25

        for i, method in enumerate(['Enhanced', 'GNN', 'Final']):
            values = []
            for metric in metrics:
                if metric in ['MAE', 'NRMSE']:
                    values.append(1 - df[f'{method}_{metric}'].mean())
                else:
                    values.append(df[f'{method}_{metric}'].mean())
            plt.bar(x + (i-1)*width, values, width, label=method)

        plt.xlabel('Metrics')
        plt.ylabel('Score (Higher is better)')
        plt.title('Performance Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'metrics_comparison.png'))
        plt.show()

  def process_all_images(self):        print(f"Processing {len(self.dataset)} images")
        model = self.train_gnn()

        for img_path in tqdm(self.dataset.image_files, desc="Processing"):
            self.process_image(img_path, model)

        self.config.metrics_df.to_csv(os.path.join(self.config.output_dir, 'metrics.csv'), index=False)
        self._plot_metrics()
        print("Pipeline completed")

if __name__ == "__main__":
    config = Config()
    pipeline = EnhancedRetinalPipeline(config)
    pipeline.process_all_images()
