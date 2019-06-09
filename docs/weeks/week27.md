---
layout: default
---
# Week 27: Temporal difference network

## Temporal difference network

I've tested a network that takes a gray scale difference image as the input image, but I've made a preprocess: 

<pre>
margin = 10
i1 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
i2 = cv2.cvtColor(imgs[i - (margin + 1)], cv2.COLOR_BGR2GRAY)
i1 = cv2.GaussianBlur(i1, (5, 5), 0)
i2 = cv2.GaussianBlur(i2, (5, 5), 0)
difference = np.zeros((i1.shape[0], i1.shape[1], 1))
difference[:, :, 0] = cv2.subtract(np.float64(i1), np.float64(i2))
mask1 = cv2.inRange(difference[:, :, 0], 15, 255)
mask2 = cv2.inRange(difference[:, :, 0], -255, -15)
mask = mask1 + mask2
difference[:, :, 0][np.where(mask == 0)] = 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
difference[:, :, 0] = cv2.morphologyEx(difference[:, :, 0], cv2.MORPH_CLOSE, kernel)
im2 = difference
if np.ptp(im2) != 0:
    img_resized = 256 * (im2 - np.min(im2)) / np.ptp(im2) - 128
else:
    img_resized = 256 * (im2 - np.min(im2)) / 1 - 128
</pre>


I've used a margin of 10 images between the 2 images. The result is: 

![dif_gray_128](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/dif_gray_128.png)

[![Follow line with Temporal difference network](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_simple_circuit.png)](https://www.youtube.com/watch?v=XkQwENb-K-Q)



|        Driving results (Temporal difference network, whole image)        |||||
|                           |          Manual         ||      Controlnet      || 
|      Circuits             |  Percentage   |   Time   | Percentage |   Time   |
|  Simple (clockwise)       |      100%     | 1min 35s |     35%    |          |
|Simple (anti-clockwise)    |      100%     | 1min 33s |     10%    |          |
|  Monaco (clockwise)       |      100%     | 1min 15s |      3%    |          | 
|Monaco (anti-clockwise)    |      100%     | 1min 15s |      3%    |          |
| Nurburgrin (clockwise)    |      100%     | 1min 02s |      8%    |          |
|Nurburgrin (anti-clockwise)|      100%     | 1min 02s |      3%    |          |


