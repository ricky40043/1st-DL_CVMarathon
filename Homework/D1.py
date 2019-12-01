# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
RGB = cv2.imread('Lena.jpg',cv2.IMREAD_COLOR )
Gray = cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE )
"""
cv2.IMREAD_COLOR (預設值)
載入包含 Blue, Green, Red 三個 channel 的彩⾊圖片
• cv2.IMREAD_GRAYSCALE
載入灰階格式的圖片
• cv2.IMREAD_UNCHANGED
載入圖片中所有 channel
"""

print( type(img))

cv2.imshow('RGB',RGB)
cv2.imshow('Gray',Gray)
cv2.waitKey(0)
cv2.destoryAllWindows()