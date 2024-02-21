# AIVision_CWD_S1OD_deployed_to_Managed_Endpoint_01
2023Nov25 - full solution - export Azure Custom Vision Model fir detecting Chronic Wasting Disease in North Amercian deer as .ONNX, production-hosted on Azure Managed Endpoint VM
this uses a predict.py local stand-alone script as the core of a score.py script Deployed as a Container to an Azure Managed Endpoint, from there it is accessed for remote inferencing
by using the Endpoint, an authorization key, and a HTTP REST POST request to send the location of a remotely-stored file (can be a locally stored file w small changes) and return
the inferenced Object Detection of infected deer (Healthy_Deer vs. UnHealthy_Deer).  bounding boxes are also returned to make it easy to identify the animals.  

This demonstration has been wound down to avoid $$$ costs.  The AML Compute was deleted, the Managed Endpoint was deleted, and the Registered Model was deleted.  the only thing remaining to use up resources is the ASA Container for remotely storing imagery data used during testing, and this AML Workspace and project

2024Feb21 To rebuild this demonstration out into working condition again:  
1) recreate Compute (default DS11_V2 or F4s_v2 will have all needed Python packages pre-installed)  
2) redownload scoreFromURLImage.py from AML project to local Downloads  
3) redownload CWD ONNX folder from AML project to local Downloads  
4) reregister CWD Model in AML 
5) redeploy model to new Endpoint in AML, D2a_v4 SKU & Pytorch 1.13 Curated Env  
6) update new Endpoint’s consume.ipynb cells with new URL, API, Model (for headers)  
7) update max_concurrent_requests_per_instance and worker_count and instance_count using CLI(v2)  
 (both = 2 for 2 core SKU, = 4 for 4 core SKU in cluster)  
8) execute test cell and repeating load test cells  
9) IF needed, copy consume.ipynb notebook & create additional Computes,   
   assigned different computes to different notebooks, execute independently for async  
   loads.   
10) Go to Azure Monitor, select Deployment then Metrics,  
    choose CPU_Util, Deployment_Cap, CPU_Memory_Util, and Disk_Util metrics
    IF needed, select only CPU_Util + apply Splitting to inspect instances loads.  



STACK STYLE HISTORY  

2024Feb21 mike retest, redeploys, retest, redeploys after Microsoft agree to provision 250 new D2a_v4 SKUs in the North Central US Region and add 32 of these to Mikes subscription quota.   all tests good.... see notes above for how to easy build out this demonstration to working condition again, using Only the contents of this AML Workspace/project file system

2024Jan23 7:29PM mike edits the github repo to make sure sync down to AML repo is also working...

2024Jan23 mcvogt MUCH later, 7:26PM, mike reclones the entire repo to AML, after wiping out ALL traces of .git files both in and above the target project folder...

2024Jan22 mcvogt more housekeeping and retesting/reverifying of all scripts  

2024Jan21 LATER after mark makes mike use the Windows App Credentials Manager (not WEB) which HAD stored michae-vogt-avanade   credentials....   got updated to 'docvogt' and mikes latest PAT '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'   

2024Jan20 mcvogt housekeeping  
updated and standardized names before final use for Microsoft Azure Learn article  
will result in only predictFromLocalImage.py predictFromURLImage.py scoreFromURLImage.py, and consume.ipynb  


