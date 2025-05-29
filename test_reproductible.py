# (Pipeline alternatif avec segmentation Frangi améliorée et analyse des seuils)
# Ce fichier propose une version améliorée du pipeline avec des étapes supplémentaires comme la normalisation des images et une carte de réponse Frangi enrichie. Il inclut une analyse des seuils Frangi pour comparer les performances et affiche des courbes F1-score en fonction des seuils.

import numpy as np
from skimage.morphology import erosion, disk, remove_small_objects, closing
from skimage.filters import frangi, threshold_otsu
from skimage import img_as_float
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm

# Élément structurant pour fermeture
se2 = np.array([[1], [0], [1]], dtype=bool)

# === Prétraitement ===
def preprocess_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return cv2.equalizeHist(img_clahe)

def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

# === Filtrage Frangi ===
def enhanced_frangi(img_float):
    scales = [frangi(img_float, sigmas=range(1, 3)),
              frangi(img_float, sigmas=range(3, 5)),
              frangi(img_float, sigmas=range(1, 5))]
    return normalize_image(np.mean(scales, axis=0))

# === Binarisation ===
def binarize_with_otsu(vessels):
    t = threshold_otsu(vessels)
    return vessels > t

def binarize_with_manual_threshold(vessels, seuil):
    return vessels > seuil

# === Nettoyage Morphologique ===
def dynamic_cleaning(img_bin, img_shape, facteur=0.0002):
    min_size = int(facteur * img_shape[0] * img_shape[1])
    return remove_small_objects(closing(img_bin, disk(1)), min_size=min_size)

# === Évaluation ===
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

# === Affichage des étapes ===
def display_step(img_step, title_base, img_GT, idx):
    precision, recall, f1 = evaluate(img_step, img_GT)
    plt.subplot(2, 3, idx)
    plt.imshow(img_step, cmap='gray')
    plt.title(f"{title_base}\nP={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    print(f"{title_base}\nP={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    plt.axis('off')

# === Pipeline complet avec visualisation ===
def my_segmentation_with_step_visualization(img, img_mask, img_GT):
    fig = plt.figure(figsize=(14, 8))

    img_clahe = preprocess_clahe(img)
    img_float = img_as_float(img_clahe)
    vessels = enhanced_frangi(img_float)

    # Visualisation carte de réponse Frangi
    plt.figure()
    plt.imshow(vessels, cmap='hot')
    plt.colorbar()
    plt.title("Carte de réponse Frangi")
    plt.show()

    imBin = binarize_with_otsu(vessels)
    imClean = dynamic_cleaning(imBin, img.shape)
    img_out = img_mask & imClean

    display_step(img_clahe, "1. CLAHE + EqualHist", img_GT, 1)
    display_step(vessels > 0, "2. Frangi (binaire large)", img_GT, 2)
    display_step(imBin, "3. Seuillage Otsu", img_GT, 3)
    display_step(imClean, "4. Nettoyage morpho", img_GT, 4)
    display_step(img_out, "5. Après masque", img_GT, 5)
    display_step(img_GT, "6. Vérité terrain", img_GT, 6)

    plt.tight_layout()
    plt.show()

    return img_out

# === Pipeline simple sans affichage ===
def my_segmentation(img, img_mask):
    img_clahe = preprocess_clahe(img)
    img_float = img_as_float(img_clahe)
    vessels = enhanced_frangi(img_float)
    imBin = binarize_with_otsu(vessels)
    imClean = dynamic_cleaning(imBin, img.shape)
    img_out = img_mask & imClean
    return img_out

# === Analyse des seuils Frangi ===
def evaluate_for_thresholds(img, img_mask, img_GT, seuils=np.linspace(0.0001, 0.05, 50)):
    Precision_list = []
    Recall_list = []
    F1_list = []

    for seuil in tqdm(seuils, desc="Seuillage Frangi pour comparaison"):
        img_clahe = preprocess_clahe(img)
        img_float = img_as_float(img_clahe)
        vessels = frangi(img_float, sigmas=range(1, 5))
        imBin = binarize_with_manual_threshold(vessels, seuil)
        imClean = dynamic_cleaning(imBin, img.shape)
        img_out = img_mask & imClean

        precision, recall, f1 = evaluate(img_out, img_GT)
        Precision_list.append(precision)
        Recall_list.append(recall)
        F1_list.append(f1)

    return Precision_list, Recall_list, F1_list

# === Courbe Precision-Recall ===
def ROC(Precision, Recall):
    plt.plot(Recall, Precision, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Precision-Recall")
    plt.grid(True)
    plt.legend()
    plt.show()

# === Chargement des données ===
img = np.asarray(Image.open('./images_IOSTAR/star02_OSC.jpg').convert('L')).astype(np.uint8)
nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(img.shape, dtype=bool)
img_mask[(row - nrows / 2)**2 + (col - ncols / 2)**2 > (nrows / 2)**2] = 0

img_GT = np.asarray(Image.open('./images_IOSTAR/GT_02.png').convert('L')).astype(np.uint8)
img_GT = img_GT > 0

# === Application du pipeline avec visualisation ===
img_out = my_segmentation_with_step_visualization(img, img_mask, img_GT)

# === Sauvegarde du résultat ===
cv2.imwrite('./images_IOSTAR/segmentation_frangi_f1.png', (img_out.astype(np.uint8) * 255))

# === Analyse seuils Frangi ===
seuils = np.linspace(0.0001, 0.05, 50)
Precision_list, Recall_list, F1_list = evaluate_for_thresholds(img, img_mask, img_GT, seuils)

# Courbe F1 en fonction du seuil
plt.plot(seuils, F1_list)
plt.xlabel("Seuil Frangi")
plt.ylabel("F1-score")
plt.title("Sensibilité du F1-score au seuil Frangi")
plt.grid(True)
plt.show()

# Affichage du meilleur seuil
best_idx = np.argmax(F1_list)
print(f"Meilleur seuil Frangi : {seuils[best_idx]:.5f} avec F1-score = {F1_list[best_idx]:.3f}")
