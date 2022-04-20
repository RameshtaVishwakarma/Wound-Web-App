from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
import numpy as np

model = load_model('./models/Model_69.h5')


def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName
    img1 = image.load_img(testimage, target_size=(150, 150))

    #plt.imshow(img1)

    Y = image.img_to_array(img1)
    #Y = Y/255
    X = np.expand_dims(Y, axis=0)

    val = model.predict(X)
    # print(val)
    if val == 1:

        context = {'filePathName': filePathName, 'predictedlabel': 'Severe'}
        return render(request, 'severe.html', context)

    elif val == 0:

        context = {'filePathName': filePathName,
                   'predictedlabel': 'Non-Severe'}
        return render(request, 'nonsevere.html', context)

    #context = {'a': 1}
    #return render(request, 'index.html', context)
    context = {'filePathName': filePathName,'predictedlabel': 'None'}
    return render(request, 'index.html', context)
    
def hospital(request):
    context = {'a': 1}
    return render(request, 'hospital.html', context)

def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)


# def predictImage(request):
#     print(request)
#     print(request.POST.dict())
#     fileObj = request.FILES['filePath']
#     fs = FileSystemStorage()
#     filePathName = fs.save(fileObj.name, fileObj)
#     filePathName = fs.url(filePathName)

#     context = {'filePathName': filePathName}
#     # if predicted_label == 'severe':
#     #     return render(request, 'index.html', context)
#     # else if predicted_label == 'non-severe':
#     #      return render(request, 'index.html', context)
#     return render(request, 'index.html', context)
