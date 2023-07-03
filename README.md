# Landing Spot Identification
Landing Spot Identification with plane detection.

Extract the LIBS.zip in the same folder as the files.

Use CMAKE to make the project and run it with the command:
./trackLANDINGPlanes ./vikon/DJI_0218.MP4 0 0.75 0 1 2 2 0 0 1 0.3 0.1 9 0.08 0.07 24 15 416 0.4

The code in "videostabKalmanLANDINGPlanes REVERSE FROM DEPTH IMAGE.cpp" can be used to emulate run of the system
from a pre-recorded depth image, that must be in a folder names images inside the folder of the executable
e.g. "images/168699_depth_norm.png"

  //EXP1 FINAL B 55
  [code]
  //./trackLANDINGPlanes ./images/168698_depth_norm.png 4 11 0 1 2 2 0 0 1 0.3 0.1 9 0.08 0.07 24 -5.00966e-03 1.4289e-02 0.003230350194553    //frame 414
  [/code]
  //EXP2 FINAL B 31
  
  //./trackLANDINGPlanes ./images/194548_depth_norm.png 1 12.55 0 1 2 2 0 0 1 0.3 0.1 9 0.08 0.07 24 -7.49339e-02 2.4893e-03 0.003230353979644    //frame 508
  
  //EXP3 FINAL B 59
  
  //./trackLANDINGPlanes ./images/173523_depth_norm.png 2 12.5 0 1 2 2 0 0 1 0.3 0.1 9 0.08 0.07 24 -6.8183e-02 1.54839e-02 0.003510403631196  //frame 600
  
  //EXP4 FINAL B 44
  
  //./trackLANDINGPlanes ./images/179259_depth_norm.png 0 12.5 0 1 2 2 0 0 1 0.3 0.1 9 0.08 0.07 24  9.14626e-03 1.06459e-03 0.003199411340924 //frame 457
