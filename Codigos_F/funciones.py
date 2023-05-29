import cv2
import numpy as np
import pandas as pd
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib.image as img
from sklearn.preprocessing import normalize                           
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from skimage import data, segmentation, color
from skimage.future import graph
from sklearn import cluster
from sklearn.cluster import MeanShift, estimate_bandwidth
from numpy import linspace
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import seaborn as sns
from cv2 import MORPH_ELLIPSE
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from kneed import KneeLocator
import seaborn
from skimage.feature import graycomatrix, graycoprops
from Codigos_F.rotating_calipers import min_max_feret 
import skimage.morphology

# diametros de feret para el objeto segmentado

def get_min_max_feret_from_mask(mask_im):
    
    eroded = skimage.morphology.erosion(mask_im)
    outline = mask_im ^ eroded
    boundary_points = np.argwhere(outline > 0)
    # convert numpy array to a list of (x,y) tuple points
    boundary_point_list = list(map(list, list(boundary_points)))
    return min_max_feret(boundary_point_list)

# SLIC
# Crea superpixeles sobre la imagen de entrada a color

def superSlic(img):
    labels1 = segmentation.slic(img,compactness=30, n_segments=185, start_label=1)
    # super EN RGB
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
    out1  = cv2.cvtColor(out1,cv2.COLOR_RGB2GRAY)
    #out1 = out1[:,:,0]
    return out1

#Kmean
#Técnica de Aprendizaje Automático no supervisado 
#que agrupa los pixeles de acuerdo  a la similaridad en su intensidad

def kmean(pixelsValues,grupos,filtro):
    #algoritmo = KMeans(grupos,init='k-means++',max_iter=300,n_init=10)
    algoritmo = KMeans(n_clusters=grupos,init = 'k-means++',random_state=0);
    algoritmo.fit(pixelsValues);
    #algoritmo.fit_predict(pixelsValues)
    centroide, etiqueta = algoritmo.cluster_centers_,algoritmo.labels_
    centroide  = np.uint8(centroide)
    segmentacion_data  = centroide[etiqueta.flatten()]
    segmentacionImagen  = segmentacion_data.reshape(filtro.shape)
    return segmentacionImagen

#Meanshift
#Algoritmo de aprendizaje no supervisado que etiqueta cada pixel de acuerdo 
#a sus caracteríticas de intensidad, y lo agrupa en un Cluster

def shift2(pixelsValues,filtro):
    bandwidth = estimate_bandwidth(pixelsValues, quantile=0.2,n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pixelsValues)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    cluster_centers = np.uint8(cluster_centers)
    segmentacionImagen = cluster_centers[labels.flatten()]
    segmentacionImagen = segmentacionImagen.reshape(filtro.shape)
    return segmentacionImagen

#Selección de Núcleos
#A partir de una imagen que contenga más de un posible nucleo,
#selecciona aquel que se encuentre mas cercano al centro de la imagen

def selContorno(contours,thresh):
    if contours == None:
        return None
    else:
        
        listaDifx = []
        listaDify = []
        dif = pd.DataFrame()

        for k in range(len(contours)):
 
            cnt = contours[k]
            

            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            dimension = thresh.shape
            refx = dimension[1]/2
            # análisis x
            if cx <= refx:
                difx= refx-cx
            else:
                difx= cx - refx
            listaDifx.append(difx)

        for k in range(len(contours)):
            cnt = contours[k]
            M = cv2.moments(cnt)
            cy = int(M['m01']/M['m00'])
            dimension = thresh.shape
            refy = dimension[0]/2
            # análisis y
            if cy <= refy:
                dify= refy-cy
            else:
                dify= cy - refy
            listaDify.append(dify)
        dif['x'] = listaDifx
        dif['y'] = listaDify
        dif['total'] = dif['x'] + dif['y']
        dif.reset_index(inplace = True, drop  = True)
        cont = dif['total'].idxmin()
        #print(dif)
        #print('contorno',cont)
        return [cont,cx,cy]

#Selección de Regiones
#Si al momento de la segmentación 
#queda el núcleo con un hueco dentro de él, esta función permite rellenar dicho núcleo

def selRegion(label,area):
    color = label.copy()
    color = color.reshape(-1,1)
    color = np.unique(color)
    color = np.sort(color,axis=None)
    color = np.uint8(color)
    for el in range(1,len(color)):
        region = label.copy()
        region[region != color[el]] = 0
        region[region == color[el]] = 255
        region = np.uint8(region)
        if np.sum(region == 255) != 0:
            contoursReg, hierarchyReg = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contornoReg,rx,ry = selContorno(contoursReg,region)
            areaReg = cv2.contourArea(contoursReg[contornoReg])
            numReg = el
            numCont = len(contoursReg)
            #print(areaReg,area)
            if areaReg == area :
                numReg = el
                numCont = len(contoursReg)
                #print(numReg,numCont)
                #numCont = len(contoursReg)
                break

    nucleoReg = label.copy()
    nucleoReg = np.uint8(nucleoReg)
    nucleoReg[nucleoReg != numReg] = 0
    nucleoReg[nucleoReg == numReg] = 255

    if numCont >=2:
        # nucleoRegFill = nucleoReg.copy()
        # h,w = nucleoReg.shape[:2]
        # mask = np.zeros((h+2,w+2),np.uint8)
        # #floofill
        # cv2.floodFill(nucleoRegFill,mask,(0,0),255)
        # #invertir
        # nucleoRegFillInv = cv2.bitwise_not(nucleoRegFill)
        # #combinar ambas imagenes
        # nucleoReg = nucleoReg | nucleoRegFillInv



        # Invertir la imagen (los agujeros se convierten en objetos)
        img_inverted = cv2.bitwise_not(nucleoReg)
        # Etiquetar los componentes conectados en la imagen
        n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(img_inverted)
        # Encuentra el componente más grande (background)
        max_label, max_area = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_components)], key=lambda x: x[1])
        # Cree una imagen en blanco con el tamaño de la imagen original
        filled_img = np.zeros(img_inverted.shape, dtype=np.uint8)
        # Rellena el componente más grande en la imagen en blanco
        filled_img[labels == max_label] = 255
        # Invertir la imagen de nuevo (los objetos se convierten en agujeros llenos)
        nucleoReg = cv2.bitwise_not(filled_img)
        
        

    return nucleoReg

#Matriz de confusión
#Cada imagen Segmentada, se la Compara con la imagen de referencia y se 
#determinan las valores de Verdadero Negativo, Verdadero Positivo, Falso Negativo y Falso Positivo, con el fin de evaluar la imagen Segmentada

def matrizConfusion (img_propuesta,img_referencia):
    #variables locales: FP falso positivo,  FN falso negativo, VP verdadero positivo, VN verdadero negativo
    FP=FN=VP=VN=0
    
    for fila in range (img_referencia.shape[0]):
        for columna in range (img_referencia.shape[1]):
            color_ref=str(img_referencia[fila,columna])
            color_pro=str(img_propuesta[fila,columna])
            if color_ref == '255' : #or color_ref == '[  0   0 255]':
                if color_pro == '255':
                    VP = VP+1
                else:
                    FN = FN+1
            if color_ref != '255': # or color_ref != '[  0   0 255]': 
                if color_pro == '255':
                    FP = FP+1
                else:
                    VN = VN+1
            
    return [FN,FP,VP,VN]

#Medidas de evaluación
#Permite evaluar la imagen según la Exactitud (E), Presición(P), Sensibilidad(R),
#Promedio Señal a Ruido (NSR), medida F1 (FM), Tasa de Error (ER) 

def MedidasMC (VP,VN,FP,FN):
    #Precisión (C) ó (P)
    vp_fp=VP+FP
    if vp_fp == 0:
        P= None
    else:
        P= VP/vp_fp

    #Recall (L)ó Revocación (R)
    if (VP+FN) == 0:
        R = None
    else:
        R= VP/(VP+FN)

    #Exactitud (E)
    if (VP+VN+FP+FN) == 0:
        E = None
    else:
        E=(VP+VN)/(VP+VN+FP+FN)
    
    #NSR El promedio señal a Ruido (*) OJO CON ESTA MEDIDA COMPROBAR 
    if VP == 0:
        NSR= None
    else:
        NSR = FP/VP

    #Medida FM ó Medida de Primer Plano
    if P is None or R is None:
        FM = None
    elif (P+R) == 0:
        FM = None
    else:
        FM = (2*P*R)/(P+R)
    

    #Tasa de Error (ER) Tasa de Falsos positivos
    if (VP+FN) == 0:
        ER= None
    else:
        ER = FP/(VP+FN)

    #Tasa Negativa (NRM), es cero cuando la segmentación es perfecta
    NR_FN = FN/(VP+FN) # tasa de Falsos negativos
    NR_FP = FP/(VP+FN) # tasa de falsos positivos

    NRM = (NR_FN + NR_FP)/2
    
    return [P,R,E,NSR,FM,ER,NRM]

# Indice de Similitud
def similitud(imagenReg,imagenRef):
    # Calcular la intersección y la unión de las imágenes
    intersection = np.logical_and(imagenRef, imagenReg)
    union = np.logical_or(imagenRef, imagenReg)
    # Calcular el índice de Jaccard
    jaccard_index = np.sum(intersection) / np.sum(union)
    dice_index = 2 * jaccard_index / (1 + jaccard_index)

    return [jaccard_index,dice_index]



