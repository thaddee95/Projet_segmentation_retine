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
IMAGE_PATH = './images_IOSTAR/star01_OSC.jpg'  # Chemin de l'image à traiter
GT_PATH = './images_IOSTAR/GT_01.png'          # Chemin de la vérité terrain
SEUILS = np.linspace(0.005, 0.05, 50)
SEED = 420

# =============== PRETRAITEMENT ===============
def preprocess_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return cv2.equalizeHist(img_clahe)

def normalize_image(img: np.ndarray) -> np.ndarray:
    return (img - np.min(img)) / (np.max(img) - np.min(img))

# =============== FILTRE DE FRANGI ===============
def enhanced_frangi(img_float: np.ndarray) -> np.ndarray:
    scales = [
        frangi(img_float, sigmas=range(1, 3)),
        frangi(img_float, sigmas=range(3, 5)),
        frangi(img_float, sigmas=range(1, 5))
    ]
    mean_frangi = np.mean(scales, axis=0)
    return normalize_image(mean_frangi)

# =============== BINARISATION ===============
def binarize_with_otsu(vessels: np.ndarray) -> np.ndarray:
    t = threshold_otsu(vessels)
    return vessels > t

def binarize_with_manual_threshold(vessels: np.ndarray, threshold: float) -> np.ndarray:
    return vessels > threshold

# =============== NETTOYAGE MORPHOLOGIQUE AMELIORE POUR OTSU SEULEMENT ===============
def dynamic_cleaning_otsu(img_bin: np.ndarray, img_shape, factor: float = 0.0002) -> np.ndarray:
    """
    Nettoyage morphologique amélioré appliqué UNIQUEMENT au pipeline Otsu :
    - fermeture plus large (disk 2) pour relier vaisseaux fins
    - ouverture classique (disk 1)
    - suppression des petits objets avec seuil plus bas
    """
    min_size = int(factor * img_shape[0] * img_shape[1])
    img_closed = closing(img_bin, disk(2))
    img_opened = opening(img_closed, disk(1))
    cleaned = remove_small_objects(img_opened, min_size=min_size)
    return cleaned

# =============== NETTOYAGE MORPHOLOGIQUE STANDARD POUR SEUIL MANUEL (pipeline best_threshold) ===============
def dynamic_cleaning_standard(img_bin: np.ndarray, img_shape, factor: float = 0.0002) -> np.ndarray:
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
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()

    # Prétraitement + filtrage Frangi
    img_clahe = preprocess_clahe(img)
    img_float = img_as_float(img_clahe)
    vessels = enhanced_frangi(img_float)

    # Pipeline Otsu modifié (nettoyage amélioré)
    t_otsu = threshold_otsu(vessels)
    im_bin_otsu = vessels > t_otsu
    im_clean_otsu = dynamic_cleaning_otsu(im_bin_otsu, img.shape)
    p_otsu_bin, r_otsu_bin, f1_otsu_bin = evaluate(im_bin_otsu, gt)
    p_otsu_clean, r_otsu_clean, f1_otsu_clean = evaluate(im_clean_otsu, gt)

    # Pipeline seuil manuel (best_threshold) avec nettoyage standard non modifié
    im_bin_best = vessels > best_threshold
    im_clean_best = dynamic_cleaning_standard(im_bin_best, img.shape)
    img_out = img_mask & im_clean_best
    p_best_bin, r_best_bin, f1_best_bin = evaluate(im_bin_best, gt)
    p_best_clean, r_best_clean, f1_best_clean = evaluate(im_clean_best, gt)
    p_best_mask, r_best_mask, f1_best_mask = evaluate(img_out, gt)

    # Affichage première ligne : pipeline Otsu
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
    axes[3].set_title(f"4. Morph. Otsu amélioré\nP={p_otsu_clean:.2f}, R={r_otsu_clean:.2f}, F1={f1_otsu_clean:.2f}")
    axes[3].axis('off')

    axes[4].imshow(gt, cmap='gray')
    axes[4].set_title("5. Vérité terrain (GT)")
    axes[4].axis('off')

    # Affichage deuxième ligne : pipeline seuil manuel
    axes[5].imshow(img_clahe, cmap='gray')
    axes[5].set_title("1. CLAHE + EqualHist")
    axes[5].axis('off')

    im6 = axes[6].imshow(vessels, cmap='hot')
    axes[6].set_title("2. Réponse Frangi")
    axes[6].axis('off')
    fig.colorbar(im6, ax=axes[6], fraction=0.046, pad=0.04)

    axes[7].imshow(im_bin_best, cmap='gray')
    axes[7].set_title(f"3. Seuil={best_threshold:.4f}\nP={p_best_bin:.2f}, R={r_best_bin:.2f}, F1={f1_best_bin:.2f}")
    axes[7].axis('off')

    axes[8].imshow(im_clean_best, cmap='gray')
    axes[8].set_title(f"4. Morph. standard\nP={p_best_clean:.2f}, R={r_best_clean:.2f}, F1={f1_best_clean:.2f}")
    axes[8].axis('off')

    axes[9].imshow(img_out, cmap='gray')
    axes[9].set_title(f"5. Masque appliqué\nP={p_best_mask:.2f}, R={r_best_mask:.2f}, F1={f1_best_mask:.2f}")
    axes[9].axis('off')

    fig.suptitle(f"Comparaison pipelines : Otsu amélioré vs Seuil optimal – Image {num}", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return img_out

# =============== SEUILS OPTIMAUX ===============
def evaluate_for_thresholds(img, img_mask, gt, thresholds=SEUILS):
    precision_list, recall_list, f1_list = [], [], []
    img_clahe = preprocess_clahe(img)
    img_float = img_as_float(img_clahe)
    vessels = enhanced_frangi(img_float)

    for threshold in tqdm(thresholds, desc="Seuils Frangi"):
        im_bin = binarize_with_manual_threshold(vessels, threshold)
        im_clean = dynamic_cleaning_standard(im_bin, img.shape)
        img_out = img_mask & im_clean
        precision, recall, f1 = evaluate(img_out, gt)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return precision_list, recall_list, f1_list

# =============== TRAITEMENT IMAGE UNIQUE ===============
def process_image(image_path, gt_path):
    img = np.asarray(Image.open(image_path).convert('L')).astype(np.uint8)
    gt = np.asarray(Image.open(gt_path).convert('L')).astype(np.uint8) > 0

    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    img_mask = np.ones(img.shape, dtype=bool)
    img_mask[(row - nrows / 2)**2 + (col - ncols / 2)**2 > (nrows / 2)**2] = 0

    precision_list, recall_list, f1_list = evaluate_for_thresholds(img, img_mask, gt, SEUILS)
    best_idx = np.argmax(f1_list)
    best_seuil = SEUILS[best_idx]

    print(f"Meilleur seuil : {best_seuil:.5f} | F1 : {f1_list[best_idx]:.3f}")

    img_out = segmentation_pipeline_with_visualization(img, img_mask, gt, best_seuil)

    output_dir = 'resultats'
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, 'segmentation.png'), (img_out.astype(np.uint8) * 255))
    
    with open(os.path.join(output_dir, 'metriques.txt'), 'w') as f:
        f.write(f"Meilleur seuil : {best_seuil:.5f}\n")
        f.write(f"Précision : {precision_list[best_idx]:.3f}\n")
        f.write(f"Rappel : {recall_list[best_idx]:.3f}\n")
        f.write(f"F1-score : {f1_list[best_idx]:.3f}\n")

np.random.seed(SEED)
process_image(IMAGE_PATH, GT_PATH)
