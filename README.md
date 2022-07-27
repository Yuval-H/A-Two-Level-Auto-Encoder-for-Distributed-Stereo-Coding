# A-Two-Level-Auto-Encoder-for-Distributed-Stereo-Coding

This is the implementation for the article "A Two-Level Auto-Encoder for Distributed Stereo Coding" (Yuval Harel and Shai Avidan, 2022)


-- link to the paper will be added --

We propose a new technique for stereo image compression that is based on Distributed Source Coding (DSC).

![Teaser](https://user-images.githubusercontent.com/76810287/181239690-dce1a22b-58f4-4780-9d8b-62d8eebda3e9.png)


### Citation
If you find our work useful in your research, please cite: (will be added)


## Abstract
We propose a new technique for stereo image compression that is based on Distributed Source Coding (DSC). In our
setting, two cameras transmit their image back to a processing unit. Naively doing so requires each camera to compress and transmit
its image independently. However, the images are correlated because they observe the same scene, and our goal is to take advantage
of this fact. In our solution, one camera, assume the left camera, sends its image to the processing unit, as before. The right camera,
on the other hand, transmits its image conditioned on the left image, even though the two cameras do not communicate. The
processing unit can then decode the right image, using the left image. The solution is based on a two level Auto-Encoder (AE). During
training, the first level AE learns a standard single image compression code. The second level AE further compresses the code of the
right image, conditioned on the code of the left image. During inference, the left camera uses the first level AE to transmit its image to
the processing unit. The right camera, on the other hand, uses the encoders of both levels to transmit its code to the processing unit.
The processing unit uses the top level decoder to recover the left image, and the decoders of both levels, as well as the recovered left
image, to recover the right image. The system achieves state of the art results in image compression on several popular datasets.


### Dataset
For training and testing were performed over *KITTI* dataset, please download files and place them in the main folder (or choose the appropriate path within the code). 
[KITTI 2012](http://www.cvlibs.net/download.php?file=data_stereo_flow_multiview.zip) and [KITTI 2015](http://www.cvlibs.net/download.php?file=data_scene_flow_multiview.zip).

It is best to run the program using your favorite IDE. 
There may be a need to modify the scripts (mainly choosing paths).

### Weights
Pre-trained model for our hardest compression point - 0.031 bit per pixel is available here (will be added)
