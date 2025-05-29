import numpy as np
import cv2
from skimage.morphology import erosion, disk, remove_small_objects, closing, opening
from skimage.filters import frangi, threshold_otsu
from skimage import img_as_float
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

# =============== CONFIGURATION ===============
IMAGE_PATH = './images_IOSTAR/star02_OSC.jpg'
GT_PATH = './images_IOSTAR/GT_02.png'
SEED = 42
SEUILS = np.linspace(0.005, 0.05, 50)

# =============== PRETRAITEMENT ===============
def preprocess_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return cv2.equalizeHist(img_clahe)

def normalize_image(img: np.ndarray) -> np.ndarray:
    return (img - np.min(img)) / (np.max(img) - np.min(img))

# =============== FILTRE DE FRANGI ===============
def enhanced_frangi(img_float: np.ndarray) -> np.ndarray:
    scales = [frangi(img_float, sigmas=range(1, 3)),
              frangi(img_float, sigmas=range(3, 5)),
              frangi(img_float, sigmas=range(1, 5))]
    return normalize_image(np.mean(scales, axis=0))

# =============== BINARISATION ===============
def binarize_with_otsu(vessels: np.ndarray) -> np.ndarray:
    t = threshold_otsu(vessels)
    return vessels > t

def binarize_with_manual_threshold(vessels: np.ndarray, threshold: float) -> np.ndarray:
    return vessels > threshold

# =============== NETTOYAGE MORPHOLOGIQUE ===============
def dynamic_cleaning(img_bin: np.ndarray, img_shape, factor: float = 0.0005) -> np.ndarray:
    min_size = int(factor * img_shape[0] * img_shape[1])
    img_closed = closing(img_bin, disk(1))
    img_opened = opening(img_closed, disk(1))
    cleaned = remove_small_objects(img_opened, min_size=min_size)
    return cleaned

# =============== EVALUATION ===============
def evaluate(seg: np.ndarray, gt: np.ndarray):
    gt_bin = gt > 0
    seg_bin = seg > 0
    TP = np.sum(gt_bin & seg_bin)
    FP = np.sum(~gt_bin & seg_bin)
    FN = np.sum(gt_bin & ~seg_bin)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# =============== VISUALISATION PIPELINE ===============
def segmentation_pipeline_with_visualization(img, img_mask, gt, best_threshold, num=''):
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))  # 2 lignes, 5 colonnes
    axes = axes.flatten()

    # Étapes communes
    img_clahe = preprocess_clahe(img)
    img_float = img_as_float(img_clahe)
    vessels = enhanced_frangi(img_float)

    # Pipeline Otsu
    im_bin_otsu = binarize_with_otsu(vessels)
    im_clean_otsu = dynamic_cleaning(im_bin_otsu, img.shape)
    p_otsu_bin, r_otsu_bin, f1_otsu_bin = evaluate(im_bin_otsu, gt)
    p_otsu_clean, r_otsu_clean, f1_otsu_clean = evaluate(im_clean_otsu, gt)

    # Pipeline seuil optimal
    im_bin_best = binarize_with_manual_threshold(vessels, best_threshold)
    im_clean_best = dynamic_cleaning(im_bin_best, img.shape)
    img_out = img_mask & im_clean_best
    p_best_bin, r_best_bin, f1_best_bin = evaluate(im_bin_best, gt)
    p_best_clean, r_best_clean, f1_best_clean = evaluate(im_clean_best, gt)
    p_best_mask, r_best_mask, f1_best_mask = evaluate(img_out, gt)

    # ---- Ligne 1 : Pipeline Otsu ----
    axes[0].imshow(img_clahe, cmap='gray')
    axes[0].set_title("1. CLAHE + EqualHist")
    axes[0].axis('off')

    im1 = axes[1].imshow(vessels, cmap='hot')
    axes[1].set_title("2. Réponse Frangi")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(im_bin_otsu, cmap='gray')
    axes[2].set_title(f"3. Binarisation Otsu\nP={p_otsu_bin:.2f}, R={r_otsu_bin:.2f}, F1={f1_otsu_bin:.2f}")
    axes[2].axis('off')

    axes[3].imshow(im_clean_otsu, cmap='gray')
    axes[3].set_title(f"4. Morph. Otsu\nP={p_otsu_clean:.2f}, R={r_otsu_clean:.2f}, F1={f1_otsu_clean:.2f}")
    axes[3].axis('off')

    axes[4].axis('off')  # Case vide pour alignement

    # ---- Ligne 2 : Pipeline Seuil optimal ----
    axes[5].imshow(img_clahe, cmap='gray')
    axes[5].set_title("1. CLAHE + EqualHist")
    axes[5].axis('off')

    im2 = axes[6].imshow(vessels, cmap='hot')
    axes[6].set_title("2. Réponse Frangi")
    axes[6].axis('off')
    fig.colorbar(im2, ax=axes[6], fraction=0.046, pad=0.04)

    axes[7].imshow(im_bin_best, cmap='gray')
    axes[7].set_title(f"3. Seuil={best_threshold:.4f}\nP={p_best_bin:.2f}, R={r_best_bin:.2f}, F1={f1_best_bin:.2f}")
    axes[7].axis('off')

    axes[8].imshow(im_clean_best, cmap='gray')
    axes[8].set_title(f"4. Morph. optimal\nP={p_best_clean:.2f}, R={r_best_clean:.2f}, F1={f1_best_clean:.2f}")
    axes[8].axis('off')

    axes[9].imshow(img_out, cmap='gray')
    axes[9].set_title(f"5. Masque appliqué\nP={p_best_mask:.2f}, R={r_best_mask:.2f}, F1={f1_best_mask:.2f}")
    axes[9].axis('off')

    fig.suptitle(f"Comparaison des pipelines : Otsu vs Seuil optimal – Image {num}", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return img_out


# =============== SEUILS OPTIMAUX ===============
def evaluate_for_thresholds(img, img_mask, gt, thresholds=SEUILS):
    precision_list, recall_list, f1_list = [], [], []
    for threshold in tqdm(thresholds, desc="Seuils Frangi"):
        img_clahe = preprocess_clahe(img)
        img_float = img_as_float(img_clahe)
        vessels = enhanced_frangi(img_float)
        im_bin = binarize_with_manual_threshold(vessels, threshold)
        im_clean = dynamic_cleaning(im_bin, img.shape)
        img_out = img_mask & im_clean
        precision, recall, f1 = evaluate(img_out, gt)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    return precision_list, recall_list, f1_list

# =============== TRAITEMENT IMAGE UNIQUE ===============
def process_single_image(gt_file, star_files, gt_dir, star_dir, seg_dir, metrics_dir, seuils):
    num = gt_file.split('_')[1].split('.')[0]
    star_candidates = [f for f in star_files if num in f]
    if not star_candidates:
        print(f"No star image found for {gt_file}")
        return
    star_file = star_candidates[0]
    print(f"Processing: {star_file} / {gt_file}")

    img = np.asarray(Image.open(os.path.join(star_dir, star_file)).convert('L')).astype(np.uint8)
    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    img_mask = np.ones(img.shape, dtype=bool)
    img_mask[(row - nrows / 2)**2 + (col - ncols / 2)**2 > (nrows / 2)**2] = 0
    gt = np.asarray(Image.open(os.path.join(gt_dir, gt_file)).convert('L')).astype(np.uint8) > 0

    precision_list, recall_list, f1_list = evaluate_for_thresholds(img, img_mask, gt, seuils)
    best_idx = np.argmax(f1_list)
    best_seuil = seuils[best_idx]

    print(f"Best threshold: {best_seuil:.5f} | F1: {f1_list[best_idx]:.3f}")

    img_out = segmentation_pipeline_with_visualization(img, img_mask, gt, best_seuil, num)

    seg_path = os.path.join(seg_dir, f'seg_{num}.png')
    cv2.imwrite(seg_path, (img_out.astype(np.uint8) * 255))

    metrics_path = os.path.join(metrics_dir, f'metrics_{num}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Best threshold: {best_seuil:.5f}\n")
        f.write(f"Precision: {precision_list[best_idx]:.3f}\n")
        f.write(f"Recall: {recall_list[best_idx]:.3f}\n")
        f.write(f"F1-score: {f1_list[best_idx]:.3f}\n")

# =============== TRAITEMENT LOT D'IMAGES ===============
def process_all_images(gt_dir='images_IOSTAR', star_dir='images_IOSTAR',
                       seg_dir='results_segmentations', metrics_dir='results_metrics'):
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.startswith('GT') and f.endswith('.png')])
    star_files = sorted([f for f in os.listdir(star_dir) if f.startswith('star') and f.endswith('.jpg')])

    for gt_file in gt_files:
        process_single_image(gt_file, star_files, gt_dir, star_dir, seg_dir, metrics_dir, SEUILS)

# =============== MAIN ===============
if __name__ == "__main__":
    np.random.seed(SEED)
    process_all_images()
