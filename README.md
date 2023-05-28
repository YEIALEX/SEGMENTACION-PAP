# Algoritmo de segmentación celular de imágenes de Papanicolaou para el apoyo diagnóstico de cáncer de cuello uterino

El análisis de frotis cervical, una técnica pionera descrita por Papanicolaou, ha
demostrado ser efectiva en la detección temprana de patologías y se utiliza
ampliamente en sistemas CAD para el diagnóstico automatizado. En este estudio, se
emplearon diversas técnicas de segmentación y clasificación de imágenes PAP con el
objetivo de apoyar el diagnóstico del cáncer de cuello uterino. Se utilizaron técnicas de
aprendizaje no supervisado, como KMEANS y MEANSHIFT, para llevar a cabo la
segmentación del núcleo celular. Estas técnicas se basan en el concepto de
agrupamiento para fusionar regiones de superpíxeles y lograr la segmentación deseada.
Además, se empleó la red UNET, un modelo de aprendizaje supervisado, para la
segmentación tanto del núcleo como del citoplasma. Para la realización de este
estudio, se utilizó la base de datos de HERLEV, que consta de 917 imágenes junto con
sus correspondientes segmentaciones y etiquetas. Esta base de datos facilitó el proceso
de aprendizaje y evaluación de las técnicas utilizadas en la investigación. En cuanto a
la clasificación, se emplearon cuatro clasificadores: DT, SVM, KNN y ANN, para
resolver problemas de clasificación binaria y de tres clases. Para mejorar el
rendimiento de la clasificación, se aplicaron técnicas como PCA y FeatureWiz para
obtener características significativas de las células. Los resultados obtenidos mostraron
que la red UNET fue altamente efectiva en la segmentación tanto del citoplasma como
del núcleo, obteniendo índices de Dice de 0.91 ± 0.044 y 0.90 ± 0.117,
respectivamente. En cuanto a la clasificación binaria, el clasificador ANN alcanzó una
exactitud del 98 %, mientras que para la clasificación de tres clases, el clasificador
KNN obtuvo una precisión del 89 %. Por tanto, este estudio demuestra la eficacia de
las técnicas de segmentación y clasificación utilizadas en el análisis de frotis cervical,
proporcionando resultados prometedores para el apoyo al diagnóstico de cáncer de
cuello uterino.

<p align="center">
  <img src="./pipe/pipeA.jpg" width="600" title="Overall Pipeline">
</p>

<p align="center">
  <img src="./pipe/pipeB.jpg" width="300" title="Overall Pipeline">
</p>

# Dataset link
1. [Herlev](http://mde-lab.aegean.gr/index.php/downloads)

Nota: base de datos disponible en el momento de realizar este proyecto.

# Intrucciones para correr el codigo

- Kmeans.
```
+-- data
+-- preprocesamientoKmean.ipynb
+-- evalNucleoKmean.csv
+-- caractNucleoKmean.csv
+-- evaluacionSegKmean.ipynb 

```
- Meanshift.
```
+-- data
+-- preprocesamientoMeanShift.ipynb
+-- evalNucleoMeanShift.csv
+-- caractNucleoShift.csv
+-- evaluacionSegShift.ipynb


```

- UNET <br>
los pesos de la red UNET deben descargarse y pegarse en sus respectivas carpetas núcleo y citoplasma <br>
[pesos UNET](https://unicaucaeduco-my.sharepoint.com/:f:/g/personal/yeinerimbachi_unicauca_edu_co/Eu-QzwGsQLFAjr8YeqswUM8BsQJPxarAX6DfmvhCaT5_XA?e=gfJJYj) 
```
+-- data
|   +-- .
|   +-- train
|   +-- val
+-- preprocesamiento.py
+-- classBin.py

+-- data
+-- preprocesamiento.ipynb
|   +-- procesamiento de las imagenes de entrada
|   +-- procesamiento imagenes de referencia
|   +-- aumento de datos
|   +-- datos de prueba
+-- entrenamientoUnet.ypinb
|   +-- .
|   +-- train
+-- evalNucleoKmean.csv
+-- caractNucleoKmean.csv
+-- evaluacionSegKmean.ipynb
+-- caractNucleoKmeanBinBal.csv
+-- classBinKmean.ypinb
+-- caractNucleoKmeanCytBal.csv
+-- classCytKmean.ipynb
+-- caractNucleoKmeanHistBal.csv
+-- classHistKmean.ipynb
+-- balance.ipynb

```


1. installar los paquetes necesarios:
```
pip install -r requirements.txt
```
Argumentos Disponibles:
Kmeans
- K=20
- n=x