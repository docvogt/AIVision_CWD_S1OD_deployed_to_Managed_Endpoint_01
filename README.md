# AIVision_CWD_S1OD_deployed_to_Managed_Endpoint_01
2023Nov25 - full solution - export Azure Custom Vision Model fir detecting Chronic Wasting Disease in North Amercian deer as .ONNX, production-hosted on Azure Managed Endpoint VM
this uses a predict.py local stand-alone script as the core of a score.py script Deployed as a Container to an Azure Managed Endpoint, from there it is accessed for remote inferencing
by using the Endpoint, an authorization key, and a HTTP REST POST request to send the location of a remotely-stored file (can be a locally stored file w small changes) and return
the inferenced Object Detection of infected deer (Healthy_Deer vs. UnHealthy_Deer).  bounding boxes are also returned to make it easy to identify the animals.  

2024Jan20 mcvogt housekeeping  
updated and standardized names before final use for Microsoft Azure Learn article  
will result in only predictFromLocalImage.py predictFromURLImage.py scoreFromURLImage.py, and consume.ipynb  


2024Jan21 LATER after mark makes mike use the Windows App Credentials Manager (not WEB) which HAD stored michae-vogt-avanade   credentials....   got updated to 'docvogt' and mikes latest PAT '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'   

2024Jan22 mcvogt more housekeeping and retesting/reverifying of all scripts  
