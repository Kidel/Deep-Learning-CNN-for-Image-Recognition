# Custom DataSets
This folder contains custom datasets.

## apple-drive-duck
This dataset was obtained sampling some videos and it contains pictures of apples, hard drives and rubber ducks. 
Those objects were chosen according to ImageNet strengths and weaknesses. 
ImageNet has a lot of pictures of apples (24 categories with thousands of images each), 
very few hard drive images (68 total) and zero rubber ducks (maybe there are collateral rubber ducks in some bathroom or toy images, but ImageNet has no rubber duck class, so the actual number is zero or close to zero, rubber ducks are classified as "baloon" by Inception Model V3). 
This should give a good variety to study how Transfer Learning behaves with new classes and old classes, also with few examples.

The set contains 551 rubber duck images (plus 155 for test), 342 hard drive images (plus 161 for test) and 537 apple images (plus 126 for test). 
For the sampled video data we used one hard drive, 2 different apples (a granny smith and a red delicious) and 2 rubber ducks (one red, one yellow). Background, lights and distance have been changed between videos and the same ones have been used for the different classes.
