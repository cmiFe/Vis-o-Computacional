#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     AJUDAR O MANEZAO DO GABRIEL
#
# Author:      guigui
#
# Created:     26/10/2020
# Copyright:   (c) guigui 2020
# Licence:     <ME DEVE 10 REAIS>
#-------------------------------------------------------------------------------
import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged

def main():
    escala = 7
    img = cv2.imread(r"C:\Users\GUI\fingers.jpg")
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    largura = int(img.shape[1] * escala / 100)
    altura = int(img.shape[0] * escala / 100)
    dim = (largura, altura)
    img_redimensionada = cv2.resize(img_cinza, dim, interpolation = cv2.INTER_AREA)
    img_redimensionada = cv2.GaussianBlur(img_redimensionada, (5, 5), 0)
    _, thresh = cv2.threshold(img_redimensionada, 165, 255, cv2.THRESH_BINARY )
    thresh = cv2.dilate(thresh, np.ones((3, 3)))
    img_canny = auto_canny(thresh)
    contornos, hierarchy = cv2.findContours(img_canny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contornos, key = cv2.contourArea)
    area = cv2.contourArea(cnt)
    lista_contornos = []
    for c in contornos:
        if cv2.contourArea(c)>=area/2:
            cv2.drawContours(img_redimensionada, [c], -1, 0, 2)
    for c in contornos:
        x,y,w,h = cv2.boundingRect(c)
        ROI = img_redimensionada[y:y+h, x:x+w]
        lista_contornos.append(ROI)

    cv2.imshow('ROI1',lista_contornos[0])
    cv2.imshow('ROI2',lista_contornos[1])
    cv2.imshow('ROI3',lista_contornos[2])
    cv2.imshow('ROI4',lista_contornos[3])
    cv2.imshow('ROI5',lista_contornos[4])
    cv2.imshow('scr2',img_redimensionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
