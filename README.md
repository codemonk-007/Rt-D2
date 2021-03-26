# Rt-D2

## Real time - Drowsiness Detection

In order to classify the state of the operator as drowsy, it is necessary for the detection system to identify the operators’ eyes on the image. As the video feed from the camera is given as an input to the system, each frame of the video is considered as an individual image .We then detect the eyes on these frames. The eye detection is performed with facial landmarks through the dlib library. The facial land mark detection is done through the HaarCascade classifiers.
One of the important landmarks related to the eye , known as the EAR or the eye aspect ratio is used to decide the current state of the operators eye.
The EAR is the proportion of of width and height of the eyes based on its landmarks. It is calculated through the given equation :

<img width="455" alt="image" src="https://user-images.githubusercontent.com/81297719/112603872-93463a80-8e3b-11eb-881b-9aaa7ab50287.png">

This EAR is calculated for each individual frame of the real time video. Thus, as we can see from the formula, that the EAR is directly proportional to the “openness” of the eye. Thus this can be used to detect the blink of an eye .

In order to detect drowsiness , we have adopted the threshold model. In this model, the a certain threshold is set to decide the blink of an eye. This threshold was set for a set of frames and then analysed. If the EAR for the set number of frames was found to be less than the provided threshold, a blaring sound is activated and the driver is alerted.

The threshold we used was 0.25 and the limiting frames used was 20. Frame of the live feed , when EAR was above threshold:

![image](https://user-images.githubusercontent.com/81297719/112603967-ad801880-8e3b-11eb-9b4c-e6b8fd6bef2b.png)

Frame of the live feed , when the EAR was below threshold :

![image](https://user-images.githubusercontent.com/81297719/112603987-b375f980-8e3b-11eb-8c9a-db3b2c1c57be.png)

When this frame is encountered the program was set to play an alarming sound to alert the operator along with an alert sound which is to be displayed on the screen.
