import numpy as np
from skimage.morphology import erosion, dilation, disk, skeletonize, remove_small_objects, reconstruction, square
from skimage.filters import frangi,sato
from skimage import img_as_float
from PIL import Image
from matplotlib import pyplot as plt
import cv2


# Éléments structurants
se1 = disk(3)
se2 = np.array([[1], [0], [1]], dtype=bool)
se3 = disk(2)
sec=square(5)

def preprocess_clahe(img):
    # Appliquer CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    return img_clahe

def my_segmentation1(img, img_mask, seuil):

    # Contraste pour mieux faire ressortir les vaisseaux
    imEro = erosion(img, sec)
    imDil = dilation(img, sec)
    imMask = ((imDil - img) < (img - imEro)) 
    imContraste = imMask*imDil + ~imMask*imEro

    # Nivellement pour supprimer le bruit de fond
    imFAS=imContraste.copy()
    taille_max = 4 # Calcul du filtre alterné séquentiel (Ouverture d'abord)
    for i in range(1,taille_max):
        se = disk(i)
        imFAS = erosion(imFAS,se)
        imFAS = dilation(imFAS,se)
        imFAS = dilation(imFAS,se)
        imFAS = erosion(imFAS,se)

    imFASmask = (imFAS > img) 
    imFASsup = imFASmask*imFAS + ~imFASmask*255 # Image partout supérieure (fond à 255)
    imFASinf = ~imFASmask*imFAS # Image partout inférieure (fond à 0)

    imFASinf_reco = reconstruction(imFASinf,img) # Reconstruction directe
    imFASsup_reco = 255 - reconstruction(255 - imFASsup,255 - img) # Reconstruction duale

    imNivel = imFASmask*imFASsup_reco + ~imFASmask*imFASinf_reco  # Recombinaison

    # Extraction des vaisseaux (gradient ou laplacien)
    imDil = dilation(imNivel,sec) # Dilatation morphologique
    imEro = erosion(imNivel, sec)
    # Gradient :
    imgradient=imDil-imEro
    # Laplacien : 
    # A=2*np.asarray(imContraste).astype(np.int16)
    # laplacien=imDil.astype(np.int16)+imEro.astype(np.int16)-A

    # Seuillage pour obtenir une image binaire
    imBin=imgradient > seuil

    # Points isolés pour supprimer du bruit
    imEro2 = erosion(imBin, se2)
    imPts1Visoles = imEro2 & imBin

    # Masque pour garder la partie étudiée
    img_out = img_mask & imPts1Visoles

    return img_out

def my_segmentation2(img, img_mask, seuil):
    # # Étape 1 : prétraitement CLAHE
    img_clahe = preprocess_clahe(img)

    # Étape 2 : filtre vasculaire sur image normalisée
    img_float = img_as_float(img_clahe)
    # Filtre de Sato
    #vessels = sato(img_float, sigmas=range(1, 3))
    # Filtre de Frangi
    vessels = frangi(img_float, sigmas=range(1, 5), scale_step=1)

    # Étape 3 : seuillage sur la sortie du filtre
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

def evaluate1(img_out, img_GT):
    # Fonction d'évaluation

    GT_skel = skeletonize(img_GT) # On reduit le support de l'evaluation...
    img_out_skel = skeletonize(img_out) # ...aux pixels des squelettes
    TP = np.sum(img_GT & img_out) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs (relaxes)
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs (relaxes)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate2(img_out, img_GT):
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
    # Si la 1ère segmentation est utilisée, remplacer seuils par np.linspace(0,40,50)
    # Si la 2ème segmentation est utilisée, remplacer seuils par np.linspace(0.001, 0.05, 50)

    best_f1_score = -1
    best_seuil = 0
    best_output = None
    Precision_list = []
    Recall_list = []
    F1_list = []

    for seuil in seuils:
        try:
            # Application de la segmentation
            img_out = my_segmentation2(img, img_mask, seuil)

            # Évaluation des performances
            precision, recall, f1 = evaluate2(img_out, img_GT)

            # Sauvegarde des scores
            Precision_list.append(precision)
            Recall_list.append(recall)
            F1_list.append(f1)

            # Mise à jour du meilleur score si nécessaire
            if f1 > best_f1_score:
                best_f1_score = f1
                best_seuil = seuil
                best_output = img_out

        except Exception as e:
            print(f"Erreur avec seuil={seuil:.4f} : {e}")
            Precision_list.append(0)
            Recall_list.append(0)
            F1_list.append(0)

    # Trouver l'indice du meilleur seuil
    best_idx = np.argmax(F1_list)

    print(f"Meilleur seuil (F1) = {seuils[best_idx]:.4f}, "
          f"Précision = {Precision_list[best_idx]:.3f}, "
          f"Rappel = {Recall_list[best_idx]:.3f}, "
          f"F1 = {F1_list[best_idx]:.3f}")

    return best_output, best_f1_score, best_seuil, Precision_list, Recall_list, F1_list

def PR(Precision, Recall):
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
    plt.title(f'Segmentation')
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
    PR(Precision_list, Recall_list)

print(f"Score F1 moyen = {np.mean(Scores):.4f}")