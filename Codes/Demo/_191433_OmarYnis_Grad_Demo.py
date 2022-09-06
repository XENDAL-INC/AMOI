import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import PySimpleGUI as sg
import os.path
import io
from PIL import Image

labels=['Bifidobacterium spp(-ve)',
 'Candida Albicans(-ve)',
 'Enterococcus faecalis(-ve)',
 'Escherichia Coli(+ve)',
 'Fusobacterium(+ve)',
 'Lactobacillus Gasseri(-ve)',
 'Listeria Monocytogenes(-ve)',
 'Pseudomonas Aeruginosa(+ve)',
 'Staphylococcus Epidermidis(-ve)',
 'Streptococcus Agalactiae(-ve)']


cnnFE = tf.keras.models.load_model("D:/Xendal/University/Grad/Models/CNN_featureExtraction.model")
ann = tf.keras.models.load_model("D:/Xendal/University/Grad/Models/ANN_Hybrid.model")
# First the window layout in 2 columns

file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Button("Predict")],
    [sg.Text(size=(40, 1), key="Prediction")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("AMOI (AI-aided Microorganisms Identification)", layout)
filename="";

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "Predict" and filename is not "":
                img = imread(filename)
                shape=min(img.shape[0], img.shape[1])
                factor=shape/225

                img = resize(img, (img.shape[0] // factor, img.shape[1] // factor), anti_aliasing='True')

                input_img = np.expand_dims(img, axis=0)
                input_img_feature=cnnFE.predict(input_img)

                input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
                prediction = ann.predict(input_img_features)[0]

                prediction_class = np.argmax(prediction)
                predBacteria = labels[prediction_class]

                window["Prediction"].update("Prediction: " + predBacteria)

    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".jpg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)

            image = Image.open(filename)
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            # Actually store the image in memory in binary 
            image.save(bio, format="PNG")
            # Use that image data in order to 
            window["-IMAGE-"].update(data=bio.getvalue())

        except:
            pass

window.close()