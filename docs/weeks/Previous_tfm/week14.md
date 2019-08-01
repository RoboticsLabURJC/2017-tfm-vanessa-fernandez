---
layout: default
---
# Week 14: Testing DetectionSuite

This week, I've been testing detectionSuite. To prove it, you need to try DatasetEvaluationApp. To check the operation of this app you have to execute: 

<pre>
./DatasetEvaluationApp --configFile=appConfig.txt
</pre>

My appConfig.txt is: 

<pre>
--datasetPath
/home/vanejessi/DeepLearningSuite/DeepLearningSuite/DatasetEvaluationApp/sampleFiles/datasets/

--evaluationsPath
/home/vanejessi/DeepLearningSuite/DeepLearningSuite/DatasetEvaluationApp/sampleFiles/evaluations

--weightsPath
/home/vanejessi/DeepLearningSuite/DeepLearningSuite/DatasetEvaluationApp/sampleFiles/weights/yolo_2017_07

--netCfgPath
/home/vanejessi/DeepLearningSuite/DeepLearningSuite/DatasetEvaluationApp/sampleFiles/cfg/darknet

--namesPath
/home/vanejessi/DeepLearningSuite/DeepLearningSuite/DatasetEvaluationApp/sampleFiles/cfg/SampleGenerator

--inferencesPath
/home/vanejessi/DeepLearningSuite/DeepLearningSuite/DatasetEvaluationApp/sampleFiles/evaluations
</pre>


 To check its operation, you have to select the following: 

![detection](https://roboticslaburjc.github.io/2017-tfm-vanessa-fernandez/images/detection.png)

However, I have had some errors in execution width CUDA:

<pre>
mask_scale: Using default '1,000000'
CUDA Error: no kernel image is available for execution on the device
CUDA Error: no kernel image is available for execution on the device: El archivo ya existe
</pre>


