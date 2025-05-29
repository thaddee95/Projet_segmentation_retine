# (Pipeline avec segmentation Frangi et fermeture pour reconnecter les vaisseaux)
# Ce fichier ajoute une étape de fermeture morphologique pour reconnecter les vaisseaux coupés après la binarisation. Il inclut également une analyse des seuils Frangi avec cette étape supplémentaire et sauvegarde les résultats.

import numpy as np
from skimage.morphology import remove_small_objects, closing
from skimage.filters import frangi, threshold_otsu
from skimage import img_as_float
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm

def preprocess_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def enhanced_frangi(img_float):
    scales = [frangi(img_float, sigmas=range(1, 3)),
              frangi(img_float, sigmas=range(3, 5)),
              frangi(img_float, sigmas=range(1, 5))]
    return normalize_image(np.mean(scales, axis=0))

def binarize_with_otsu(vessels):
    t = threshold_otsu(vessels)
    return vessels > t

def reconnect_vessels(binary_img):
    # Élément structurant orienté en forme de croix diagonale pour reconnecter les vaisseaux coupés
    selem = np.array([[0, 1, 1, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 1, 1, 1, 0]], dtype=bool)
    return closing(binary_img, selem)

def dynamic_cleaning(img_bin, img_shape, facteur=0.0005):
    min_size = int(facteur * img_shape[0] * img_shape[1])
    return remove_small_objects(img_bin, min_size=min_size)

def evaluate(img_out, img_GT):
    GT_bin = img_GT > 0
    out_bin = img_out > 0
    TP = np.sum(GT_bin & out_bin)
    FP = np.sum(~GT_bin & out_bin)
    FN = np.sum(GT_bin & ~out_bin)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def display_step(img_step, title_base, img_GT, idx):
    precision, recall, f1 = evaluate(img_step, img_GT)
    plt.subplot(2, 3, idx)
    plt.imshow(img_step, cmap='gray')
    plt.title(f"{title_base}\nP={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    print(f"{title_base}\nP={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    plt.axis('off')

def my_segmentation_with_step_visualization(img, img_mask, img_GT):
    fig = plt.figure(figsize=(14, 8))

    img_clahe = preprocess_clahe(img)
    img_float = img_as_float(img_clahe)
    vessels = enhanced_frangi(img_float)
    imBin = binarize_with_otsu(vessels)
    imReconnected = reconnect_vessels(imBin)
    imClean = dynamic_cleaning(imReconnected, img.shape)
    img_out = img_mask & imClean

    display_step(img_clahe, "1. CLAHE", img_GT, 1)
    display_step(vessels > 0, "2. Frangi (binaire)", img_GT, 2)
    display_step(imBin, "3. Seuillage Otsu", img_GT, 3)
    display_step(imReconnected, "4. Fermeture reconnect.", img_GT, 4)
    display_step(imClean, "5. Nettoyage morpho", img_GT, 5)
    display_step(img_out, "6. Après masque", img_GT, 6)

    plt.tight_layout()
    plt.show()

    return img_out

def my_segmentation(img, img_mask):
    img_clahe = preprocess_clahe(img)
    img_float = img_as_float(img_clahe)
    vessels = enhanced_frangi(img_float)
    imBin = binarize_with_otsu(vessels)
    imReconnected = reconnect_vessels(imBin)
    imClean = dynamic_cleaning(imReconnected, img.shape)
    img_out = img_mask & imClean
    return img_out

def evaluate_for_thresholds(img, img_mask, img_GT, seuils=np.linspace(0.0001, 0.05, 50)):
    Precision_list = []
    Recall_list = []
    F1_list = []

    for seuil in tqdm(seuils, desc="Seuillage Frangi pour comparaison"):
        img_clahe = preprocess_clahe(img)
        img_float = img_as_float(img_clahe)
        vessels = frangi(img_float, sigmas=range(1, 5))
        imBin = vessels > seuil
        imReconnected = reconnect_vessels(imBin)
        imClean = dynamic_cleaning(imReconnected, img.shape)
        img_out = img_mask & imClean

        precision, recall, f1 = evaluate(img_out, img_GT)
        Precision_list.append(precision)
        Recall_list.append(recall)
        F1_list.append(f1)

    return Precision_list, Recall_list, F1_list

def ROC(Precision, Recall):
    plt.plot(Recall, Precision, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Precision-Recall")
    plt.grid(True)
    plt.legend()
    plt.show()

# Chargement de l'image et du masque
img = np.asarray(Image.open('./images_IOSTAR/star32_ODC.jpg').convert('L')).astype(np.uint8)
nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(img.shape, dtype=bool)
img_mask[(row - nrows / 2)**2 + (col - ncols / 2)**2 > (nrows / 2)**2] = 0

# Vérité terrain
img_GT = np.asarray(Image.open('./images_IOSTAR/GT_32.png').convert('L')).astype(np.uint8)
img_GT = img_GT > 0

# Application segmentation complète avec visualisation étape par étape
img_out = my_segmentation_with_step_visualization(img, img_mask, img_GT)

# Sauvegarde du résultat
cv2.imwrite('./images_IOSTAR/segmentation_frangi_f1_reconnected.png', (img_out.astype(np.uint8) * 255))

# Évaluation sur plages de seuils Frangi pour voir sensibilité (avec fermeture)
_, _, F1_list = evaluate_for_thresholds(img, img_mask, img_GT)
plt.plot(np.linspace(0.0001, 0.05, 50), F1_list)
plt.xlabel("Seuil Frangi")
plt.ylabel("F1-score")
plt.title("Sensibilité du F1-score au seuil Frangi (avec fermeture)")
plt.grid(True)
plt.show()
