# MNDNR_CWDS1OD_Deployed_to_Managed_Endpoint_01
# 2023Nov25 - full solution - export Azure Custom Vision Model fir detecting Chronic Wasting Disease in North Amercian deer as .ONNX, production-hosted on Azure Managed Endpoint VM
# this uses a predict.py local stand-alone script as the core of a score.py script Deployed as a Container to an Azure Managed Endpoint, from there it is accessed for remote inferencing
# by using the Endpoint, an authorization key, and a HTTP REST POST request to send the location of a remotely-stored file (can be a locally stored file w small changes) and return
# the inferenced Object Detection of infected deer (Healthy_Deer vs. UnHealthy_Deer).  bounding boxes are also returned to make it easy to identify the animals.  
