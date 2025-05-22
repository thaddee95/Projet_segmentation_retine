import numpy as np
from skimage.morphology import erosion, disk, skeletonize, remove_small_objects
from skimage.filters import frangi
from skimage import img_as_float
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm

# Éléments structurants
se2 = np.array([[1], [0], [1]], dtype=bool)

def preprocess_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

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

def my_segmentation_with_step_visualization(img, img_mask, seuil, img_GT):
    fig = plt.figure(figsize=(14, 8))

    img_clahe = preprocess_clahe(img)
    img_float = img_as_float(img_clahe)
    vessels = frangi(img_float, sigmas=range(1, 5), scale_step=1)
    imBin = vessels > seuil
    imEro2 = erosion(imBin, se2)
    imPts1Visoles = imEro2 & imBin
    imClean = remove_small_objects(imPts1Visoles, min_size=100)
    img_out = img_mask & imClean

    display_step(img_clahe, "1. CLAHE", img_GT, 1)
    display_step(vessels > 0, "2. Frangi (binaire)", img_GT, 2)
    display_step(imBin, f"3. Seuillage (> {seuil:.4f})", img_GT, 3)
    display_step(imClean, "4. Nettoyage morpho", img_GT, 4)
    display_step(img_out, "5. Après masque", img_GT, 5)
    display_step(img_GT, "6. Vérité terrain", img_GT, 6)

    plt.tight_layout()
    plt.show()

    return img_out

def my_segmentation(img, img_mask, seuil):
    img_clahe = preprocess_clahe(img)
    img_float = img_as_float(img_clahe)
    vessels = frangi(img_float, sigmas=range(1, 5), scale_step=1)
    imBin = vessels > seuil
    imEro2 = erosion(imBin, se2)
    imPts1Visoles = imEro2 & imBin
    imClean = remove_small_objects(imPts1Visoles, min_size=100)
    img_out = img_mask & imClean
    return img_out

def find_best_threshold(img, img_mask, img_GT, seuils=np.linspace(0.0001, 0.05, 50)):
    best_score = -1
    best_seuil = 0
    best_out = None
    Precision_list = []
    Recall_list = []
    F1_list = []

    for seuil in tqdm(seuils, desc="Recherche du meilleur seuil"):
        img_out = my_segmentation(img, img_mask, seuil)
        precision, recall, f1 = evaluate(img_out, img_GT)
        Precision_list.append(precision)
        Recall_list.append(recall)
        F1_list.append(f1)

        if f1 > best_score:
            best_score = f1
            best_seuil = seuil
            best_out = img_out

    print(f"Meilleur seuil (F1) = {best_seuil:.4f}, "
          f"P = {Precision_list[np.argmax(F1_list)]:.3f}, "
          f"R = {Recall_list[np.argmax(F1_list)]:.3f}, "
          f"F1 = {best_score:.3f}")

    # Tracer F1-score en fonction du seuil
    plt.plot(seuils, F1_list, label='F1-score')
    plt.xlabel('Seuil')
    plt.ylabel('F1-score')
    plt.title('Évolution du F1-score selon le seuil')
    plt.grid(True)
    plt.legend()
    plt.show()

    return best_out, best_seuil, Precision_list, Recall_list, F1_list

def ROC(Precision, Recall):
    plt.plot(Recall, Precision, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Precision-Recall")
    plt.grid(True)
    plt.legend()
    plt.show()

img = np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg').convert('L')).astype(np.uint8)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(img.shape, dtype=bool)
img_mask[(row - nrows / 2)**2 + (col - ncols / 2)**2 > (nrows / 2)**2] = 0

img_GT = np.asarray(Image.open('./images_IOSTAR/GT_01.png').convert('L')).astype(np.uint8)
img_GT = img_GT > 0

# Recherche du meilleur seuil
img_out, best_seuil, Precision_list, Recall_list, F1_list = find_best_threshold(img, img_mask, img_GT)

# Affichage détaillé avec visualisation
_ = my_segmentation_with_step_visualization(img, img_mask, best_seuil, img_GT)

# Sauvegarde du résultat
cv2.imwrite('./images_IOSTAR/segmentation_frangi_f1.png', (img_out.astype(np.uint8) * 255))

# Courbe Precision-Recall
ROC(Precision_list, Recall_list)
