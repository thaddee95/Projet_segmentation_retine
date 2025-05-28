import numpy as np
from skimage.morphology import erosion, dilation, disk, skeletonize, remove_small_objects
from skimage.filters import frangi,sato
from skimage import img_as_float
from PIL import Image
from matplotlib import pyplot as plt
import cv2


# Éléments structurants
se1 = disk(3)
se2 = np.array([[1], [0], [1]], dtype=bool)
se3 = disk(2)

def preprocess_clahe(img):
    # Appliquer CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    return img_clahe

def my_segmentation(img, img_mask, seuil):
    # # Étape 1 : prétraitement CLAHE
    img_clahe = preprocess_clahe(img)

    # Étape 2 : filtre de Sato sur image normalisée
    img_float = img_as_float(img_clahe)
    vessels = sato(img_float, sigmas=range(1, 3))

    # Étape 3 : seuillage sur la sortie Sato
    imBin = vessels > seuil

    # Étape 4 : nettoyage morphologique
    imEro2 = erosion(imBin, se2)
    imPts1Visoles = imEro2 & imBin
    imClean = remove_small_objects(imPts1Visoles, min_size=100)

    # Appliquer le masque du fond d’œil
    img_out = img_mask & imClean

    # Fermeture pour faire disparaître les petites structures blanches
    img_out = erosion(dilation(img_out, se3))

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

    return best_out, best_score, best_seuil, Precision_list, Recall_list, F1_list

def ROC(Precision, Recall):
    Ref=[]
    X=np.linspace(0,1,len(Recall))
    for x in X:
        Ref.append(1-x)
    plt.plot(Recall, Precision, label="Courbe Precision-Recall")
    plt.plot(X, Ref, '--')
    plt.title("Courbe Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()


# --- Liste des images de travail ---

Img_tests=['./images_IOSTAR/star01_OSC.jpg',
           './images_IOSTAR/star02_OSC.jpg',
           './images_IOSTAR/star03_OSN.jpg',
           './images_IOSTAR/star08_OSN.jpg',
           './images_IOSTAR/star21_OSC.jpg',
           './images_IOSTAR/star26_ODC.jpg',
           './images_IOSTAR/star28_ODN.jpg',
           './images_IOSTAR/star32_ODC.jpg',
           './images_IOSTAR/star37_ODN.jpg',
           './images_IOSTAR/star48_OSN.jpg']

Img_verite=['./images_IOSTAR/GT_01.png',
            './images_IOSTAR/GT_02.png',
            './images_IOSTAR/GT_03.png',
            './images_IOSTAR/GT_08.png',
            './images_IOSTAR/GT_21.png',
            './images_IOSTAR/GT_26.png',
            './images_IOSTAR/GT_28.png',
            './images_IOSTAR/GT_32.png',
            './images_IOSTAR/GT_37.png',
            './images_IOSTAR/GT_48.png']

Img_result=['./images_IOSTAR/segmentation_01.png',
            './images_IOSTAR/segmentation_02.png',
            './images_IOSTAR/segmentation_03.png',
            './images_IOSTAR/segmentation_08.png',
            './images_IOSTAR/segmentation_21.png',
            './images_IOSTAR/segmentation_26.png',
            './images_IOSTAR/segmentation_28.png',
            './images_IOSTAR/segmentation_32.png',
            './images_IOSTAR/segmentation_37.png',
            './images_IOSTAR/segmentation_48.png']


# --- Segmentation sur une base de 10 images ---

# Liste des f1 scores de chaque image
Scores=[]

for i in range(len(Img_tests)):

    # --- Chargement image et vérité terrain ---
    img = np.asarray(Image.open(Img_tests[i]).convert('L')).astype(np.uint8)
    print(f"Image shape: {img.shape}")

    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    img_mask = np.ones(img.shape, dtype=bool)
    invalid_pixels = ((row - nrows / 2) ** 2 + (col - ncols / 2) ** 2 > (nrows / 2) ** 2)
    img_mask[invalid_pixels] = 0

    img_GT = np.asarray(Image.open(Img_verite[i]).convert('L')).astype(np.uint8)
    img_GT = img_GT > 0  # binariser GT

    # Trouver le meilleur seuil (max F1)
    img_out, best_score, best_seuil, Precision_list, Recall_list, F1_list = find_best_threshold(img, img_mask, img_GT)
    Scores.append(best_score)

    # Sauvegarde et affichage
    cv2.imwrite(Img_result[i], (img_out.astype(np.uint8) * 255))

    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Image Originale')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(img_out, cmap='gray')
    plt.title(f'Segmentation (Sato, seuil={best_seuil:.4f})')
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

print(f"Score F1 moyen = {np.mean(Scores):.4f}")