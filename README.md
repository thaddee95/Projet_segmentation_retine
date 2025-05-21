# Cours MI206 : Projet segmentation rétine

L'ophtalmoscopie à balayage laser (dite SLO, pour scanning laser ophthalmoscopy) est une modalité
d'imagerie de la rétine permettant de réaliser un fond d'oeil a grande résolution et avec un large champ,
et donc d'observer sur une seule image la majeure partie de la surface de la rétine à une résolution entre
10 et 100 micromètres.  

Outre les maladies de la rétine elle-même, l'observation du fond d'oeil permet de diagnostiquer plusieurs
pathologies générales en observant la circulation artérielle et veineuse dans la rétine. C'est le cas en
particulier de l'hypertension artérielle et de l'insuffisance rénale. Le diagnostic repose en général sur
une analyse quantitative de l'ensemble du réseau vasculaire de l'image de rétine, et nécessite donc une
segmentation précise de ce réseau.  

Le but de ce projet est de proposer une méthode automatique de segmentation du réseau vasculaire
dans des images de rétine SLO. Les images de vérité terrain (Ground Truth) ont été annotées manuellement par un expert et servent de référence. 
