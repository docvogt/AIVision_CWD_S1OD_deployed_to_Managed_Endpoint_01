# 2023Nov from   https://stackoverflow.com/questions/59175701/score-py-azureml-for-images
# this is the COMMENT-lite version...
# this file was archived after creating a successful Deployment to a Managed Endpoint using a Pytorch 1.13 Curated Envrionment and 3X E4s_v3 SKU w 4cores/32GBRAM/64GBSSD
# this was updated to include a slew of GOF print statements to instrument the log file and help debugging a dreadfully slow bult-test cycle (~1 hr turnaround time)
# after 7th, corrected use of PIL.Image, and inserted extra print()'s 

# notes from inspecting the README.md file exported from CUV as part of the .ONNX model format...
# Custom Vision Export Object Detection Models
# This model is exported from Custom Vision Service

# Please visit our Sample scripts respository.

# Prerequisites
# (For TensorFlow Lite model) TensorFlow Lite 2.1 or newer

# Input specification
# This model expects 320x320, 3-channel RGB images. Pixel values need to be in the range of [0-255].# <-- we can hard-code these dimensions....

# Output specification
# There are three outputs from this model.

# detected_boxes The detected bounding boxes. Each bounding box is represented as [x1, y1, x2, y2] where (x1, y1) and (x2, y2) are the coordinates of box corners.
# detected_scores Probability for each detected boxes.
# detected_classes The class index for the detected boxes.



#--------------- HISTORY STACK ------------------
# HISTORY - push DOWN stack-fashion...

# 2023Nov20 mcvogt
# recovering/relocating from East-island Internet failure..    interrupted my first attemot to Deploy 021c...   trying again from T-Mobile hotspoit in West side of island (St Thomas USVI)...

# ...
# much development in weeks between Oct and end of 2023Nov
# ...

# THIS will be scoreURL015.py
# this will convert the incoming 2nd argument from being a filepath in the local file system, to using a URL to a remotely-stored file
# it will be based on the fully-functional and tested predictFromURL.py (orginally called predictFromURI.py, but 'I" is tough to read...)
# when using local version of predict.py, "./TestImage.jpg" was the local test file.
# now, when using predictFromURL.py, "https://asatestimages01.blob.core.windows.net/container-testimages/DEER_CWD_imagery/Cropped_IC_Model/DeerDayHealthyCandidates/TestImageBlob.jpg"
# will be the test image, stored with public access in Mike's Azure Storage Account



#---------------- start of code -----------------
#---------------- start of code -----------------
#---------------- start of code -----------------
# IMPORTS
import json   # so, json is already here, so will have json.dumps() available...

import os

import PIL.Image  # <--- this is the way predict.py works....

import requests   # mike left this off in 015

from io import BytesIO
from azureml.contrib.services.aml_request import AMLRequest, rawhttp   # this provides for the @rawhttp so dont need flask...
from azureml.contrib.services.aml_response import AMLResponse

import onnx
import onnxruntime

import numpy as np
from numpy import array     # from numpy, import specific data types so this file's code matches other code within onnx and onnxruntime
from numpy import float32   # from numpy, import specific data types so this file's code matches other code within onnx and onnxruntime
from numpy import int64     # from numpy, import specific data types so this file's code matches other code within onnx and onnxruntime

import time
import sys
from azureml.core.model import Model
#----------------------------------------------------

#----------------------------------------------------
# CONSTANTS
PROB_THRESHOLD = 0.01  # Minimum probably to show results., this is to match predict.py
#----------------------------------------------------

#-------------------------------------------------------------------
class Model:
    def __init__(self, model_filepath):
        # self.session = onnxruntime.InferenceSession(str(model_filepath)) # creating a session was done in score's init(), where session is made global...
        # will create a session using the tested Endpoint Deployment code, but will DO this in the location where predict.py did it - during model creation...
        # create an inferencing session - again, should be a one-time thing and persist...
        self.session = onnxruntime.InferenceSession(
    #       modelPath, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # will leave in place for when CUDA/GPU might be available...    
            modelPath, providers=['CPUExecutionProvider']  # this only gives option for CPUs SKUs... not GPUs....  
        )
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True
    #------------------------------------------------------------------------------------------------------------------------------
    def predict(self, image_filepath):  # when this is called by main(), args.image_filepath becomes local image_filepath...
        url = image_filepath  # this is local handle to the image's filepathname...
        image = PIL.Image.open(requests.get(url, stream=True).raw) # NOTE - this is version of PIL.Image.open() for urls, not local files!!
        image = image.resize(self.input_shape)  # this works only inside predict.py because class Model is defined here...

        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}
#-------------------------------------------------------------------



#------------ helper functions ------------------
# https://www.freecodecamp.org/news/python-switch-statement-switch-case-example/
# NOTE - for the helper tagging function, these class tag definitions need to be edited to match the labels.txt of the users model.   the CWD model only
#        had the two categories - Healthy and Unhealthy...   this helper function only makes the inferencing output easier to read, does Not affect inferencing.  
def switch(class_id):
    if class_id == 0:
        return "Healthy Deer"
    elif class_id == 1:
        return "UnHealthy Deer"

def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            clabel = switch(class_id)
#           print(f"Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
            print(f"Label: {clabel}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
#------------ helper functions ------------------


# NOTE - init() and run() are the two Required functions inside of score.py for a Managed Endpoint...
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def init():
    print("entering init()")
    global modelPath # NOTE - this model path is set at development time, NOT run time, it's fixed, unchanging...  NOT read in as an argument.  
    global model     # if going to create model here, make it available to run() and helper functions??
    # global session   # 

    # go fetch modelPath from Deployment's Container....
    modelPath = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'NADeerCWDobjectdetector.ONNX/model.onnx')   # obtained from Deployment Container...                   
    print("modelPath: ", modelPath) # for debugging, print model path to log so can inspect it post-run...

    # create a [global] model from the modelPath...  should be a one-time thing, not each time user inferences an image file...
    # model = Model(args.model_filepath)            # this creates a model using the model source location argument... this also creates a session...
    model = Model(modelPath)            # this creates a model using the model source location argument... this also creates a session...
    print("created object model of Model class inside score's init()...")
# ------------------------------------------------------------
    print("creating model object also created a model.session...")
    print("exiting init()")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------
@rawhttp   # what exactly does this do?????
def run(request):         # this SHOULD be repeated w every image processed.  extract image_url from POST request, execute model.predict using the new image_url, print results, return outputs
    print("entering run()...")
    if request.method == 'POST':
        print("entering POST request handler loop...")
        dictData = request.get_json()  # extract ALL of JSON key-value pairs in the data payload, into a Dictionary...
        print("extracting dictData from POST request = ", dictData) # for debugging...
        image_url = dictData["image_url"] # look-up or extract the one image_url value from the 'image_url' key
        print("extracting image_url value using key = ", image_url) # for debugging...

        # print("run() executed by with model.predict() commented out, NO actual inferencing !!!!!")  #<-----------------------------------------------------------------------------
        # outputs = "dummy placeholder outputs so can comment out actual inferencing!!!!"             #<-- this is NOT real outputs, just allows complete execution and return value..

        try:
            # outputs = model.predict(args.image_filepath)  # this uses the created model and calls predict method passing in the image source URL argument...
            outputs = model.predict(image_url)  # this uses the created model and calls predict method passing in the image source URL argument...
            print("called model.predict() inside of run()...")

            print("outputs is of type: ", type(outputs))
            print("[raw] outputs = ", outputs)
            print_outputs(outputs) # outputs are returned by model.predict(), but Not printed - call helper function to print them w tags.  prints to LOG!!!!!!!!
            #------------------

            # the following block of code prepares 'outputs' for json serialization, needed for return to POST request... 
            outputsFL = {} # initialize outputs Flattened, List version of ndarray
            outputsFL["detected_boxes"] = outputs["detected_boxes"].flatten().tolist()
            outputsFL["detected_classes"] = outputs["detected_classes"].flatten().tolist()
            outputsFL["detected_scores"] = outputs["detected_scores"].flatten().tolist()
            json_outputsFL = json.dumps(outputsFL)
            print("json serialization of outputsFL = ", json_outputsFL)

            #------------------
        except Exception as ex:
            # handle the exception
            outputs = "exception thrown for model.predict(). see error below."
            print(ex)

        print("done with if POST loop...")
        print("Exiting run()...")

        # return outputs # use the outputs from predict.py's version of calling model.predict() 
        # return {"outputs": outputs}  #<-- this forcing to be a dictionary structure may be/is likely redundant, as we did verify that 'outputs' IS of type dict... 
        # return json.dumps(outputs)    # <-- this did not work in 021b either... same persistent error...
        return json_outputsFL    # <-- this should BE the Flattened, Listed, json-serialzied outputs from model.predict()...
    else:
        return AMLResponse("bad request, use POST", 500)
#---------------------------------------------------------------------------------------



# ------------------- end of code -----------------------
# ------------------- end of code -----------------------
# ------------------- end of code -----------------------
