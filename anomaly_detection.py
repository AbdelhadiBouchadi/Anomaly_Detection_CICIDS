"""
Network Traffic Anomaly Detection
Dataset: CICIDS 2017/2018 from Kaggle
Approaches: Supervised (Random Forest, XGBoost) vs Unsupervised (Isolation Forest, Autoencoder)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Core ML ──────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import xgboost as xgb

# ── Deep Learning ─────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

# ── Visualization ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import joblib
import os


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

class CICIDSPreprocessor:
    """
    Handles loading and cleaning of CICIDS 2017/2018 dataset.
    The dataset contains labeled network flows with attack types.
    """

    # CICIDS key feature groups
    FLOW_FEATURES = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
        'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
        'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
        'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
        'Avg Bwd Segment Size', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
        'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
        'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ]

    LABEL_COL = 'Label'

    def __init__(self, sample_size: int = 200_000):
        self.sample_size = sample_size
        self.scaler = RobustScaler()       # Robust to outliers in network data
        self.label_encoder = LabelEncoder()
        self.feature_cols = None

    def load(self, path: str) -> pd.DataFrame:
        """Load one or more CSV files (glob or single path)."""
        import glob
        files = glob.glob(path) if '*' in path else [path]
        dfs = []
        for f in files:
            print(f"  Loading {os.path.basename(f)}...")
            df = pd.read_csv(f, low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and sanitize CICIDS data."""
        # Replace inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columns with >40% missing
        thresh = len(df) * 0.6
        df.dropna(axis=1, thresh=thresh, inplace=True)

        # Fill remaining NaN with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # Clip extreme values (99.9th percentile)
        for col in num_cols:
            if col != self.LABEL_COL:
                upper = df[col].quantile(0.999)
                df[col] = df[col].clip(upper=upper)

        
        # Normalize label column name variations
        label_variants = ['Label', 'label', ' Label', 'label ', 'Attack Type', 'attack_type', 'Class']
        for v in label_variants:
            if v in df.columns:
                df.rename(columns={v: self.LABEL_COL}, inplace=True)
                break

        return df

    def preprocess(self, df: pd.DataFrame):
        """Full preprocessing → returns X, y (binary), y_multi, feature_names."""
        df = self.clean(df)

        # Identify available feature columns
        available = [c for c in self.FLOW_FEATURES if c in df.columns]
        if len(available) < 10:
            # Fall back: use all numeric columns except label
            available = [c for c in df.select_dtypes(include=[np.number]).columns
                         if c != self.LABEL_COL]
        self.feature_cols = available
        print(f"  Using {len(self.feature_cols)} features")

        X = df[self.feature_cols].values

        
        # Force everything to string and uppercase to be perfectly safe
        y_multi_raw = df[self.LABEL_COL].astype(str).str.strip().str.upper()
        
        print("\n" + "!"*50)
        print("DIAGNOSTIC: Here are the top labels actually inside your file:")
        print(y_multi_raw.value_counts().head(10))
        print("!"*50 + "\n")

        y_multi = self.label_encoder.fit_transform(y_multi_raw)

        # Catch every possible variation of "Normal" traffic
        normal_labels = ['BENIGN', 'NORMAL', 'NORMAL TRAFFIC', '0', '0.0', 'NAN', 'NONE', 'NULL', 'REGULAR']
        y_binary = (~y_multi_raw.isin(normal_labels)).astype(int).values
        

        # Sample if large
        if len(X) > self.sample_size:
            idx = np.random.choice(len(X), self.sample_size, replace=False)
            X, y_binary, y_multi = X[idx], y_binary[idx], y_multi[idx]

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y_binary, y_multi, self.feature_cols

    def get_class_dist(self, df: pd.DataFrame) -> pd.Series:
        return df[self.LABEL_COL].str.strip().value_counts()


# ═════════════════════════════════════════════════════════════════════════════
# 2. SUPERVISED MODELS
# ═════════════════════════════════════════════════════════════════════════════

class SupervisedDetector:
    """Random Forest + XGBoost for labeled anomaly detection."""

    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        self.xgb = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42
        )
        self.results = {}

    def train_evaluate(self, X_train, X_test, y_train, y_test, feature_names):
        """Train both models and collect metrics."""
        models = {'Random Forest': self.rf, 'XGBoost': self.xgb}

        for name, model in models.items():
            print(f"\n  Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            self.results[name] = {
                'accuracy':  accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall':    recall_score(y_test, y_pred, zero_division=0),
                'f1':        f1_score(y_test, y_pred, zero_division=0),
                'roc_auc':   roc_auc_score(y_test, y_prob),
                'avg_precision': average_precision_score(y_test, y_prob),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'y_pred': y_pred,
                'y_prob': y_prob,
                'feature_importance': (
                    pd.Series(model.feature_importances_, index=feature_names)
                    .sort_values(ascending=False).head(20)
                ),
                'report': classification_report(y_test, y_pred,
                           target_names=['BENIGN', 'ATTACK'])
            }
            print(f"    F1={self.results[name]['f1']:.4f}  "
                  f"ROC-AUC={self.results[name]['roc_auc']:.4f}  "
                  f"Recall={self.results[name]['recall']:.4f}")

        return self.results


# ═════════════════════════════════════════════════════════════════════════════
# 3. UNSUPERVISED MODELS
# ═════════════════════════════════════════════════════════════════════════════

class UnsupervisedDetector:
    """Isolation Forest + Autoencoder for unsupervised anomaly detection."""

    def __init__(self):
        self.iso_forest = IsolationForest(
            n_estimators=200,
            contamination=0.1,     # Assume ~10% anomalies
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=42
        )
        self.autoencoder = None
        self.ae_threshold = None
        self.results = {}

    # ── Autoencoder architecture ──────────────────────────────────────────────
    def _build_autoencoder(self, input_dim: int) -> Model:
        enc_dim = max(8, input_dim // 4)
        bottleneck = max(4, input_dim // 16)

        inputs = layers.Input(shape=(input_dim,))
        # Encoder
        x = layers.Dense(enc_dim, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(enc_dim // 2, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        encoded = layers.Dense(bottleneck, activation='relu', name='bottleneck')(x)
        # Decoder
        x = layers.Dense(enc_dim // 2, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(enc_dim, activation='relu')(x)
        decoded = layers.Dense(input_dim, activation='linear')(x)

        model = Model(inputs, decoded, name='Autoencoder')
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
        return model

    def train_iso_forest(self, X_train, X_test, y_test):
        print("\n  Training Isolation Forest...")
        # Train on ALL data (unsupervised — no labels used)
        self.iso_forest.fit(X_train)
        scores = -self.iso_forest.score_samples(X_test)  # Higher = more anomalous
        preds = (self.iso_forest.predict(X_test) == -1).astype(int)

        self.results['Isolation Forest'] = self._metrics(y_test, preds, scores, 'Isolation Forest')
        return self.results['Isolation Forest']

    def train_autoencoder(self, X_train, X_test, y_test):
        print("\n  Training Autoencoder...")
        input_dim = X_train.shape[1]
        self.autoencoder = self._build_autoencoder(input_dim)

        # Train only on BENIGN traffic (indices where y_test is 0 in train split)
        # In a real scenario, we train on assumed-clean data
        cb = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=50, batch_size=512,
            validation_split=0.1,
            callbacks=cb, verbose=0
        )

        # Reconstruction error as anomaly score
        recon = self.autoencoder.predict(X_test, verbose=0)
        mse_scores = np.mean(np.square(X_test - recon), axis=1)

        # Threshold: 95th percentile of training reconstruction error
        train_recon = self.autoencoder.predict(X_train, verbose=0)
        train_mse = np.mean(np.square(X_train - train_recon), axis=1)
        self.ae_threshold = np.percentile(train_mse, 95)

        preds = (mse_scores > self.ae_threshold).astype(int)
        self.results['Autoencoder'] = self._metrics(y_test, preds, mse_scores, 'Autoencoder')
        self.results['Autoencoder']['history'] = history.history
        self.results['Autoencoder']['threshold'] = self.ae_threshold
        return self.results['Autoencoder']

    def _metrics(self, y_true, y_pred, scores, name):
        m = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_true, scores),
            'avg_precision': average_precision_score(y_true, scores),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'y_pred': y_pred,
            'y_prob': scores,
        }
        print(f"    F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}  Recall={m['recall']:.4f}")
        return m


# ═════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

class ResultVisualizer:
    PALETTE = ['#00d4ff', '#ff6b6b', '#ffd93d', '#6bcb77']
    BG = '#0d1117'
    FG = '#e6edf3'
    GRID = '#21262d'

    def _style(self):
        plt.rcParams.update({
            'figure.facecolor': self.BG, 'axes.facecolor': self.BG,
            'axes.edgecolor': self.GRID, 'axes.labelcolor': self.FG,
            'xtick.color': self.FG, 'ytick.color': self.FG,
            'text.color': self.FG, 'grid.color': self.GRID,
            'grid.alpha': 0.4, 'font.family': 'monospace'
        })

    def comparison_dashboard(self, sup_results, unsup_results, save_path):
        self._style()
        all_results = {**sup_results, **unsup_results}
        models = list(all_results.keys())
        colors = dict(zip(models, self.PALETTE))

        fig = plt.figure(figsize=(20, 14), facecolor=self.BG)
        fig.suptitle('Network Anomaly Detection — Model Comparison\nCICIDS 2017/2018',
                     fontsize=16, color=self.FG, fontweight='bold', y=0.98)

        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

        # ── Bar chart: metrics ────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :2])
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        x = np.arange(len(metrics))
        w = 0.18
        for i, (name, res) in enumerate(all_results.items()):
            vals = [res[m] for m in metrics]
            bars = ax1.bar(x + i * w, vals, w, label=name,
                           color=colors[name], alpha=0.85, edgecolor='none')
            for bar, val in zip(bars, vals):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f'{val:.2f}', ha='center', va='bottom',
                         fontsize=6.5, color=self.FG)
        ax1.set_xticks(x + w * 1.5)
        ax1.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=8)
        ax1.set_ylim(0, 1.12)
        ax1.set_title('Performance Metrics', color=self.FG, fontsize=11)
        ax1.legend(fontsize=8, framealpha=0.2)
        ax1.grid(axis='y', alpha=0.3)

        # ── Radar chart ───────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2:], polar=True)
        cats = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        N = len(cats)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        for name, res in all_results.items():
            vals = [res[m] for m in metrics] + [res[metrics[0]]]
            ax2.plot(angles, vals, 'o-', color=colors[name], linewidth=2, label=name)
            ax2.fill(angles, vals, color=colors[name], alpha=0.1)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(cats, size=8, color=self.FG)
        ax2.set_ylim(0, 1)
        ax2.set_title('Radar Overview', color=self.FG, fontsize=11, pad=15)
        ax2.tick_params(colors=self.FG)
        ax2.grid(color=self.GRID, alpha=0.5)
        ax2.set_facecolor(self.BG)
        ax2.spines['polar'].set_color(self.GRID)

        # ── Confusion matrices ────────────────────────────────────────────────
        cmap = LinearSegmentedColormap.from_list('cyber', ['#0d1117', '#00d4ff'])
        for i, (name, res) in enumerate(all_results.items()):
            ax = fig.add_subplot(gs[1, i])
            cm = res['confusion_matrix']
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap,
                        ax=ax, linewidths=0.5, linecolor=self.GRID,
                        cbar=False, annot_kws={'size': 9})
            ax.set_title(f'{name}\nConfusion Matrix', color=self.FG, fontsize=9)
            ax.set_xlabel('Predicted', color=self.FG, fontsize=8)
            ax.set_ylabel('Actual', color=self.FG, fontsize=8)
            ax.set_xticklabels(['BENIGN', 'ATTACK'], fontsize=7)
            ax.set_yticklabels(['BENIGN', 'ATTACK'], fontsize=7, rotation=0)

        # ── Feature importance (RF) ───────────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, :2])
        if 'Random Forest' in sup_results:
            fi = sup_results['Random Forest']['feature_importance'].head(15)
            bars = ax5.barh(fi.index[::-1], fi.values[::-1],
                            color=self.PALETTE[0], alpha=0.8)
            ax5.set_title('Top 15 Features — Random Forest', color=self.FG, fontsize=10)
            ax5.set_xlabel('Importance', color=self.FG, fontsize=8)
            ax5.grid(axis='x', alpha=0.3)
            ax5.tick_params(labelsize=7)

        # ── AE training loss ──────────────────────────────────────────────────
        ax6 = fig.add_subplot(gs[2, 2:])
        if 'Autoencoder' in unsup_results and 'history' in unsup_results['Autoencoder']:
            h = unsup_results['Autoencoder']['history']
            ax6.plot(h['loss'], color=self.PALETTE[2], label='Train Loss', linewidth=2)
            ax6.plot(h['val_loss'], color=self.PALETTE[1], label='Val Loss',
                     linewidth=2, linestyle='--')
            ax6.set_title('Autoencoder Training Loss', color=self.FG, fontsize=10)
            ax6.set_xlabel('Epoch', color=self.FG, fontsize=8)
            ax6.set_ylabel('MSE', color=self.FG, fontsize=8)
            ax6.legend(fontsize=8, framealpha=0.2)
            ax6.grid(alpha=0.3)

        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=self.BG, edgecolor='none')
        plt.close()
        print(f"\n  Dashboard saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(data_path: str, output_dir: str = 'outputs'):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("  CICIDS Network Anomaly Detection Pipeline")
    print("=" * 60)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    print("\n[1/4] Loading & preprocessing data...")
    prep = CICIDSPreprocessor(sample_size=150_000)
    df_raw = prep.load(data_path)
    print(f"  Raw shape: {df_raw.shape}")
    X, y_binary, y_multi, feature_names = prep.preprocess(df_raw)
    print(f"  Preprocessed: {X.shape}  Attacks: {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")

    # Train/test split — stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # ── 2. Supervised ─────────────────────────────────────────────────────────
    print("\n[2/4] Supervised models (Random Forest + XGBoost)...")
    sup = SupervisedDetector()
    sup_results = sup.train_evaluate(X_train, X_test, y_train, y_test, feature_names)

    # ── 3. Unsupervised ───────────────────────────────────────────────────────
    print("\n[3/4] Unsupervised models (Isolation Forest + Autoencoder)...")
    unsup = UnsupervisedDetector()
    unsup.train_iso_forest(X_train, X_test, y_test)
    unsup.train_autoencoder(X_train, X_test, y_test)
    unsup_results = unsup.results

    # ── 4. Visualize ──────────────────────────────────────────────────────────
    print("\n[4/4] Generating dashboard...")
    viz = ResultVisualizer()
    viz.comparison_dashboard(sup_results, unsup_results,
                             os.path.join(output_dir, 'comparison_dashboard.png'))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    all_res = {**sup_results, **unsup_results}
    header = f"{'Model':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
    print(header)
    print("-" * len(header))
    for name, r in all_res.items():
        print(f"{name:<20} {r['accuracy']:>6.4f} {r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} {r['f1']:>6.4f} {r['roc_auc']:>6.4f}")

    # Save models
    joblib.dump(sup.rf,  os.path.join(output_dir, 'rf_model.pkl'))
    joblib.dump(sup.xgb, os.path.join(output_dir, 'xgb_model.pkl'))
    joblib.dump(unsup.iso_forest, os.path.join(output_dir, 'iso_forest.pkl'))
    if unsup.autoencoder:
        unsup.autoencoder.save(os.path.join(output_dir, 'autoencoder.h5'))
    joblib.dump(prep.scaler, os.path.join(output_dir, 'scaler.pkl'))
    print("\n  Models saved to outputs/")

    return all_res


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/*.csv'
    run_pipeline(path)
