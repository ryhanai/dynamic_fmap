import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize


tsne = TSNE(
    n_components=3,
    perplexity=40,         # n に応じて 20-60 程度で調整
    init="pca",
    learning_rate="auto",
    n_iter=1000,
    random_state=0,
    verbose=0
)


def gen_data(n_samples=800):
    # ===== 1) 連続軌道の 32D 特徴量と連続ラベル y を作成 =====
    y = np.linspace(0.0, 1.0, n_samples)              # 0→1 の連続ラベル（色に使う）
    t = 4.0 * np.pi * y                        # 軌道パラメータ（0→4π）

    # 32D 連続軌道（多周波 sin/cos の組合せ + わずかなノイズ）
    feat_list = []
    K = 16                                    # sin/cos で 2K = 32 次元
    for k in range(1, K + 1):
        feat_list.append(np.sin(k * t))
        feat_list.append(np.cos(k * t))
        X = np.stack(feat_list, axis=1).astype(np.float32)  # (n, 32)

    rng = np.random.default_rng(0)
    X += 0.03 * rng.standard_normal(X.shape)  # 軽い観測ノイズ
    return X, y
    

def fit(X):
    # ===== 2) 前処理 + t-SNE で 3D に圧縮 =====
    X_std = StandardScaler().fit_transform(X)
    return tsne.fit_transform(X_std)  # (n, 3)

    
def plot(trajectories, task='StackCube-v1', draw_colorbar=True, draw_axis_labels=False):
    X = np.concatenate(trajectories, axis=0)
    y = np.concatenate([np.linspace(0.0, 1.0,traj.shape[0]) for traj in trajectories])

    l = fit(X)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    norm = Normalize(vmin=0.0, vmax=1.0)
    sc = ax.scatter(l[:, 0], l[:, 1], l[:, 2],
                    c=y, cmap='viridis', norm=norm, s=10, alpha=0.95)
    

    if draw_colorbar:
        cb = plt.colorbar(sc, ax=ax, pad=0.1)
        cb.set_label('t (0 → 1)')

    if draw_axis_labels:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    ax.set_title(f'{task} (point force embedding)')
    plt.tight_layout()
    plt.show()


trajectories = np.load('emb.npy', allow_pickle=True)

res = plot(trajectories)
