# Landing Spot Identification
Landing Spot Identification with plane detection.

Extract the LIBS.zip in the same folder as the files.

Use CMAKE to make the project and run it with the command:
./trackLANDINGPlanes ./vikon/DJI_0218.MP4 0 0.75 0 1 2 2 0 0 1 0.3 0.1 9 0.08 0.07 24 15 416 0.4

The code in "videostabKalmanLANDINGPlanes REVERSE FROM DEPTH IMAGE.cpp" can be used to emulate run of the system
from a pre-recorded depth image, that must be in a folder names images inside the folder of the executable
e.g. "images/168699_depth_norm.png"
