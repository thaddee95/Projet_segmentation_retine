import numpy as np
from skimage.morphology import erosion, dilation, disk, skeletonize, remove_small_objects
from skimage.filters import frangi
from skimage import img_as_float
from PIL import Image
from matplotlib import pyplot as plt
import cv2

# Éléments structurants
se1 = disk(3)
se2 = np.array([[1], [0], [1]], dtype=bool)

# def preprocess_clahe(img):
#     # Appliquer CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     img_clahe = clahe.apply(img)
#     return img_clahe

def my_segmentation(img, img_mask, seuil):
    # # Étape 1 : prétraitement CLAHE
    # img_clahe = preprocess_clahe(img)

    # Étape 2 : filtre de Frangi sur image normalisée
    img_float = img_as_float(img)
    vessels = frangi(img_float, sigmas=range(1, 5), scale_step=1)

    # Étape 3 : seuillage sur la sortie Frangi
    imBin = vessels > seuil

    # Étape 4 : nettoyage morphologique
    imEro2 = erosion(imBin, se2)
    imPts1Visoles = imEro2 & imBin
    imClean = remove_small_objects(imPts1Visoles, min_size=100)

    # Appliquer le masque du fond d’œil
    img_out = img_mask & imPts1Visoles

    return img_out

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

def find_best_threshold(img, img_mask, img_GT, seuils=np.linspace(0.001, 0.05, 50)):
    best_score = -1
    best_seuil = 0
    best_out = None
    Precision_list = []
    Recall_list = []
    F1_list = []

    for seuil in seuils:
        img_out = my_segmentation(img, img_mask, seuil)
        precision, recall, f1 = evaluate(img_out, img_GT)
        Precision_list.append(precision)
        Recall_list.append(recall)
        F1_list.append(f1)

        if f1 > best_score:
            best_score = f1
            best_seuil = seuil
            best_out = img_out

    print(f"Meilleur seuil (F1) = {best_seuil:.4f}, Precision = {Precision_list[np.argmax(F1_list)]:.3f}, Recall = {Recall_list[np.argmax(F1_list)]:.3f}, F1 = {best_score:.3f}")

    return best_out, best_seuil, Precision_list, Recall_list, F1_list

def ROC(Precision, Recall):
    plt.plot(Recall, Precision, label="Courbe Precision-Recall")
    plt.title("Courbe Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

# --- Chargement image et vérité terrain ---
img = np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg').convert('L')).astype(np.uint8)
print(f"Image shape: {img.shape}")

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(img.shape, dtype=bool)
invalid_pixels = ((row - nrows / 2) ** 2 + (col - ncols / 2) ** 2 > (nrows / 2) ** 2)
img_mask[invalid_pixels] = 0

img_GT = np.asarray(Image.open('./images_IOSTAR/GT_01.png').convert('L')).astype(np.uint8)
img_GT = img_GT > 0  # binariser GT

# Trouver le meilleur seuil (max F1)
img_out, best_seuil, Precision_list, Recall_list, F1_list = find_best_threshold(img, img_mask, img_GT)

# Sauvegarde et affichage
cv2.imwrite('./images_IOSTAR/segmentation_frangi_f1.png', (img_out.astype(np.uint8) * 255))

plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Image Originale')
plt.axis('off')

plt.subplot(232)
plt.imshow(img_out, cmap='gray')
plt.title(f'Segmentation (Frangi, seuil={best_seuil:.4f})')
plt.axis('off')

plt.subplot(233)
plt.imshow(skeletonize(img_out), cmap='gray')
plt.title('Segmentation Squelette')
plt.axis('off')

plt.subplot(235)
plt.imshow(img_GT, cmap='gray')
plt.title('Vérité Terrain')
plt.axis('off')

plt.subplot(236)
plt.imshow(skeletonize(img_GT), cmap='gray')
plt.title('Vérité Terrain Squelette')
plt.axis('off')

plt.tight_layout()
plt.show()

# Tracer la courbe Precision-Recall
ROC(Precision_list, Recall_list)
