import random
import os
import pandas as pd
import cv2

from PIL import Image


if __package__ is None or __package__ == '':
    from Functions import *
    from COMPONENT_LEVELS import COMPONENT_LEVELS
    
else:
    from .Functions import *
    from .COMPONENT_LEVELS import COMPONENT_LEVELS

def insertAnnotation(filename, xmin, ymin, xmax, ymax, clazz):
    return (
        filename,
        clazz,
        int(xmin),
        int(ymin),
        int(xmax),
        int(ymax),
    )

def screenshot2Sketch(imagePath, outputPath, fileName):
    
    objects, lvl = [], 0
    image = openImageRGB(imagePath)

    for level in COMPONENT_LEVELS:
        objects.append([])
        elements = level.keys()
        for element in elements:
            color = level[element]
            binaryImage = binaryMaskedImage(image, color)
            contours = getContours(binaryImage)
            for contour in contours:
                x, y, w, h = getBoundingRectPoints(contour)
                objects[lvl].append({element: [x, y, x+w, y+h]})
        lvl += 1
    
    sketchImage = getBlankImage(image.shape)

    lvl = 1
    bold = False
    xml_list = []
    for level in objects:
        for element in level:
            key = list(element.keys())[0]
            xmin, ymin, xmax, ymax = element[key]
            insertSketch(sketchImage, key, (xmin, ymin, xmax, ymax), bold)
            xml_list.append(insertAnnotation(fileName + '.jpg', xmin, ymin, xmax, ymax, key))
        lvl += 1
        
    cv2.imwrite(outputPath + fileName + '.jpg', sketchImage)
    column_name = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    if os.path.isfile(outputPath + 'labels.csv'):
        xml_df.to_csv(outputPath + 'labels.csv', mode='a', index=None, header=False)
    else:
        xml_df.to_csv(outputPath + 'labels.csv', index=None)
    print('done:', fileName)