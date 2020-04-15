import cv2
import tensorflow.keras.models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_core.python.keras import models

from license_plate_extraction import segment_characters
from plate_detection import extract_plate

img1=cv2.imread("aa.jpg")
img2, img=extract_plate(img1)
char = segment_characters(img)

# %%
try:
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(char[i], cmap='gray')
        plt.axis('off')
# the code that can cause the error
except IndexError: # catch the error
    pass


# %%
"""
### Model for characters
"""

# %%
newmodel=models.load_model("model.h5")
def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28))
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        y_ = newmodel.predict_classes(img)[0] #predicting the class
        character = dic[y_] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

print(show_results())

# %%
plt.figure(figsize=(10,6))
for i,ch in enumerate(char):
    img = cv2.resize(ch, (28,28))
    plt.subplot(3,4,i+1)
    plt.imshow(img,cmap='gray')
    plt.title(f'predicted: {show_results()[i]}')
    plt.axis('off')
plt.show()

# %%
