import numpy as np
import cv2
from skimage.morphology import disk, remove_small_objects, closing, opening
from skimage.filters import frangi, threshold_otsu
from skimage import img_as_float
from PIL import Image
from matplotlib import pyplot as plt

def my_segmentation3(image_path, seuils=np.linspace(0.005, 0.05, 50), seed=42):
    np.random.seed(seed)
    img = np.asarray(Image.open(image_path).convert('L')).astype(np.uint8)
    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    img_mask = ((row - nrows / 2)**2 + (col - ncols / 2)**2 <= (nrows / 2)**2)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = cv2.equalizeHist(clahe.apply(img))
    img_float = img_as_float(img_clahe)

    vessels = np.mean([
        frangi(img_float, sigmas=range(1, 3)),
        frangi(img_float, sigmas=range(3, 5)),
        frangi(img_float, sigmas=range(1, 5))
    ], axis=0)
    vessels = (vessels - vessels.min()) / (vessels.max() - vessels.min())

    def morph_clean(img_bin, closing_disk_size=2, factor=0.0002):
        min_size = int(factor * nrows * ncols)
        return remove_small_objects(
            opening(closing(img_bin, disk(closing_disk_size)), disk(1)),
            min_size=min_size
        )

    # Masques binaires bruts pour seuils manuels
    masks_manuel_brut = [vessels > seuil for seuil in seuils]
    # Masques nettoyés manuels
    masks_manuel = [img_mask & morph_clean(m, closing_disk_size=1) for m in masks_manuel_brut]

    # Masque brut et nettoyé Otsu
    otsu_threshold = threshold_otsu(vessels)
    mask_otsu_brut = vessels > otsu_threshold
    mask_otsu = img_mask & morph_clean(mask_otsu_brut, closing_disk_size=2)

    return {
        'img': img,
        'img_clahe': img_clahe,
        'vessels': vessels,
        'mask_otsu_brut': mask_otsu_brut,
        'mask_otsu': mask_otsu,
        'masks_manuel_brut': masks_manuel_brut,
        'masks_manuel': masks_manuel,
        'seuils': seuils,
        'otsu_threshold': otsu_threshold,
        'img_mask': img_mask,
    }

def evaluate_segmentation(mask, gt):
    gt_bin, mask_bin = gt > 0, mask > 0
    TP = np.sum(gt_bin & mask_bin)
    FP = np.sum(~gt_bin & mask_bin)
    FN = np.sum(gt_bin & ~mask_bin)
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    return (precision, recall, 2 * precision * recall / (precision + recall)) if precision + recall else (precision, recall, 0)

def plot_results(data, gt_path):
    gt = np.asarray(Image.open(gt_path).convert('L')).astype(bool)
    img_clahe = data['img_clahe']
    vessels = data['vessels']

    mask_otsu_brut = data['mask_otsu_brut']
    mask_otsu = data['mask_otsu']

    masks_manuel_brut = data['masks_manuel_brut']
    masks_manuel = data['masks_manuel']
    seuils = data['seuils']
    img_mask = data['img_mask']

    # Évaluation masques manuels nettoyés
    metrics = [evaluate_segmentation(mask, gt) for mask in masks_manuel]
    best_idx = np.argmax([m[2] for m in metrics])
    best_seuil = seuils[best_idx]
    best_mask = masks_manuel[best_idx]
    best_metrics = metrics[best_idx]

    # Évaluation masques manuels bruts
    metrics_brut = [evaluate_segmentation(mask, gt) for mask in masks_manuel_brut]
    best_mask_brut = masks_manuel_brut[best_idx]
    best_metrics_brut = metrics_brut[best_idx]

    # Évaluation masque Otsu brut et nettoyé
    metrics_otsu_brut = evaluate_segmentation(mask_otsu_brut, gt)
    metrics_otsu = evaluate_segmentation(mask_otsu, gt)

    print(f"Meilleur seuil manuel : {best_seuil:.5f} avec F1={best_metrics[2]:.3f}")
    print(f"Seuil Otsu : {data['otsu_threshold']:.5f} avec F1={metrics_otsu[2]:.3f}")

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()

    # Ligne 1 : Otsu
    axes[0].imshow(img_clahe, cmap='gray')
    axes[0].set_title("1. CLAHE + EqualHist")
    axes[0].axis('off')

    im1 = axes[1].imshow(vessels, cmap='hot')
    axes[1].set_title("2. Réponse Frangi")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(mask_otsu_brut, cmap='gray')
    axes[2].set_title(f"3. Masque Otsu brut\nP={metrics_otsu_brut[0]:.2f}, R={metrics_otsu_brut[1]:.2f}, F1={metrics_otsu_brut[2]:.2f}")
    axes[2].axis('off')

    axes[3].imshow(mask_otsu, cmap='gray')
    axes[3].set_title(f"4. Masque Otsu nettoyé\nP={metrics_otsu[0]:.2f}, R={metrics_otsu[1]:.2f}, F1={metrics_otsu[2]:.2f}")
    axes[3].axis('off')

    axes[4].imshow(gt, cmap='gray')
    axes[4].set_title("5. Vérité terrain (GT)")
    axes[4].axis('off')

    # Ligne 2 : meilleur seuil manuel
    axes[5].imshow(img_clahe, cmap='gray')
    axes[5].set_title("1. CLAHE + EqualHist")
    axes[5].axis('off')

    im6 = axes[6].imshow(vessels, cmap='hot')
    axes[6].set_title("2. Réponse Frangi")
    axes[6].axis('off')
    fig.colorbar(im6, ax=axes[6], fraction=0.046, pad=0.04)

    axes[7].imshow(best_mask_brut, cmap='gray')
    axes[7].set_title(f"3. Seuil manuel brut={best_seuil:.4f}\nP={best_metrics_brut[0]:.2f}, R={best_metrics_brut[1]:.2f}, F1={best_metrics_brut[2]:.2f}")
    axes[7].axis('off')

    axes[8].imshow(best_mask, cmap='gray')
    axes[8].set_title(f"4. Seuil manuel nettoyé\nP={best_metrics[0]:.2f}, R={best_metrics[1]:.2f}, F1={best_metrics[2]:.2f}")
    axes[8].axis('off')

    axes[9].imshow(best_mask & img_mask, cmap='gray')
    axes[9].set_title(f"5. Masque circulaire appliqué\nP={best_metrics[0]:.2f}, R={best_metrics[1]:.2f}, F1={best_metrics[2]:.2f}")
    axes[9].axis('off')

    fig.suptitle("Comparaison segmentation : Otsu vs Seuil manuel optimal", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Exemple d'appel (modifie les chemins si besoin)
data = my_segmentation3('./images_IOSTAR/star01_OSC.jpg')
plot_results(data, './images_IOSTAR/GT_01.png')
