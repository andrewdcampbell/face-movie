# face-movie 
Create a video warp sequence of human faces. Can be used, for example, to create a time-lapse video showing someone's face change over time. See demos [here](https://www.youtube.com/watch?v=sbHCar2T-e0) and [here](TODO).

Supported on Python 3 and OpenCV 3+. Tested on macOS High Sierra.

## Requirements
* OpenCV
  * If you're using [Anaconda](https://www.anaconda.com/download/), run `conda install -c menpo opencv`.
* Dlib
  * If you're using [Anaconda](https://www.anaconda.com/download/), run `conda install -c menpo dlib`.  
* Boost and Boost-Python
  * If you're using [Anaconda](https://www.anaconda.com/download/), run `conda install -c meznom boost-python`.
* scipy
* numpy
* matplotlib
* PIL
* ffmpeg
  * If you're using [Homebrew](https://brew.sh), run `brew install ffmpeg`.

## Installation
1. Clone the repo.
```
git clone https://github.com/andrewdcampbell/face-movie
```
2. Download the trained face detector model from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Unzip it and place it in the root directory of the repo.

## Creating a face movie - reccomended workflow

1) Make a directory `<FACE_MOVIE_DIR>` in the root directory of the repo with the desired face images. The images must feature a clear frontal view of the desired face (other faces can be present too). The image filenames must be in lexicographic order of the order in which they are to appear in the video.

2) Create a directory `<ALIGN_OUTPUT>`. Then align the faces in the images with 
```
python face-movie/align.py -images <FACE_MOVIE_DIR> -target <BASE_IMAGE> [-overlay] [-border <BORDER>] -outdir <ALIGN_OUTPUT>
```
The output will be saved to the provided `<ALIGN_OUTPUT>` directory. BASE_IMAGE is the image to which all other images will be aligned to. It should represent the "typical" image of all your images - it will determine the output dimensions and facial position.

The optional `-overlay` flag places subsequent images on top of each other (recommended). The optional `-border <BORDER>` argument adds a white border `<BORDER>` pixels across to all the images for aesthetics. I think around 5 pixels looks good.

If your images contain multiple faces, a window will appear with the faces annotated and you will be prompted to enter the index of the correct face on the command line. 

At this point you should inspect the output images and re-run the alignment with new parameters until you're satisfied with the result.

3) Morph the sequence with 
```
python main.py -morph -images <ALIGN_OUTPUT> -td <TRANSITION_DUR> -pd <PAUSE_DUR> -fps <FPS> -out <OUTPUT_NAME>.mp4
```
This will create a video `OUTPUT_NAME.mp4` in the root directory with the desired parameters. Note that `TRANSITION_DUR` and `PAUSE_DUR` are floating point values while `FPS` is an integer. 

You may again be prompted to choose the correct face.

4) (Optional) Add music with
```
ffmpeg -i <OUTPUT_NAME>.mp4 -i <AUDIO_TRACK> -map 0:v -map 1:a -c copy -shortest -pix_fmt yuv420p <NEW_OUTPUT_NAME>.mov
```

The images used to produce the first demo video are included in `demos/daniel`. Step 2 should produce something like the aligned images shown in `demos/daniel_aligned`. Step 3 should produce something like the video `daniel.mp4`.

### Notes

* If you have photos that are already the same size and all of the faces are more or less aligned already (e.g. roster photos or yearbook photos), you can skip steps 1 and 2 and just use the directory of photos as your aligned images.


## Averaging Faces

You can also use the code to create a face average. Follow the same steps 1) - 2) as above. You probably don't want to overlay images or use a border, however. Then run
```
python face-movie/main.py -average -images <ALIGN_OUTPUT> -out <OUTPUT_NAME>.jpg
```

A small face dataset is included in the demos directory.

<img src="https://github.com/andrewdcampbell/face-movie/blob/master/demos/face_dataset/male_faces.png" width="600"> 
<img src="https://github.com/andrewdcampbell/face-movie/blob/master/demos/face_dataset/female_faces.png" width="600">

The computed average male and female face are shown below.

<img src="https://github.com/andrewdcampbell/face-movie/blob/master/demos/male_avg.jpg" width="320"> <img src="https://github.com/andrewdcampbell/face-movie/blob/master/demos/female_avg.jpg" width="320">

## Acknowledgements
* Facial landmark and image alignment code adapted from https://matthewearl.github.io/2015/07/28/switching-eds-with-python/.
* ffmpeg command adapted from https://github.com/kubricio/face-morphing.
* Affine transform code adapted from https://www.learnopencv.com/face-morph-using-opencv-cpp-python/.
