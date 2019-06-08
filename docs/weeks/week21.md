---
layout: default
---
# Week 21: Driving videos, Stacked network

## Driving videos

### Stacked network

I've used the predictions of the stacked (pilotnet with stacked frames) network (regression network) to driving a formula 1: 

[![Follow line with Stacked network for w and v (Dataset 3, test1)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_simple_circuit.png)](https://www.youtube.com/watch?v=4UfAGb1jT8Q)


I've used the predictions of the stacked (pilotnet with stacked frames) network (regression network) for w and constant v to driving a formula 1: 

[![Follow line with Stacked network for w and constant v (Dataset 3, test1)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_simple_circuit.png)](https://www.youtube.com/watch?v=4QJjUSFDcbY)


### Pilotnet network

I've used the predictions of the pilotnet network (regression network) to driving a formula 1 (test2): 

[![Follow line with Pilotnet network for w and v (Dataset 3, test2, Monaco)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_monaco.png)](https://www.youtube.com/watch?v=_NNvZ3ju7Ek)


[![Follow line with Pilotnet network for w and v (Dataset 3, test2, Nurburgrin)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_nurburgrin.png)](https://www.youtube.com/watch?v=EiTnOc2-D9Y)


In the following video complete one lap (simulation time: 1 min 26s): 

[![Follow line with Pilotnet network for w and v (Dataset 3, test2, Monaco2)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_monaco.png)](https://www.youtube.com/watch?v=Kp_BOd2uQCo)


 I've used the predictions of the pilotnet network (regression network) for w and constant v to driving a formula 1: 

[![Follow line with Pilotnet network for w and constant v (Dataset 3, test1, Monaco)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_monaco.png)](https://www.youtube.com/watch?v=9MT-rwaxFW8)


[![Follow line with Pilotnet network for w and constant v (Dataset 3, test1, Nurburgrin)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_nurburgrin.png)](https://www.youtube.com/watch?v=_KS54-g6yz0)


### Biased classification network

I've used the predictions of the classification network according to w (7 classes) and constant v to driving a formula 1: 

[![Follow line with classification network for w and constant v (Dataset 3, test2, biased, Nurburgrin)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_nurburgrin.png)](https://www.youtube.com/watch?v=ffhaT1Uvd-E)


[![Follow line with classification network for w and constant v (Dataset 3, test2, biased, Monaco)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_monaco.png)](https://www.youtube.com/watch?v=099L-bGYwXg)



## Stacked network

In this method (stacked frames), we concatenate multiple subsequent input images to create a stacked image. Then, we feed this stacked image to the network as a single input. We refer to this method as stacked. This means that for image it at time/frame t, images it−1, it−2, ... will be concatenated. In our case, we have stacked 3 images separated by 2 frames. This means that for image it at time / frame t we concatenate the image t, t-3 and t-6. 

