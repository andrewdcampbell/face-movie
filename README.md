To create a face movie:
================================

1) Make a directory <FACE_MOVIE_DIR> with the desired face images. The images must feature a clear frontal view of the desired face (other faces are okay). The image filenames must be such that when sorted, they appear in the desired order in the video.

2) Create a directory <ALIGN_OUTPUT>. Then align the images with `python align.py -images <FACE_MOVIE_DIR> -target <FACE_MOVIE_DIR>/<BASE_IMAGE> [-overlay] [-border <BORDER_IN_PIXELS>] -outdir <ALIGN_OUTPUT>`.

You may be prompted to enter the index of the correct face if multiple faces are detected.

The output will be saved to the provided <ALIGN_OUTPUT> directory. BASE_IMAGE is the image to which all other images will be aligned to. It should represent the "typical" image of all your images - it will determine the output dimensions and facial position.

The overlay flag places subsequent images on top of each other. I recommend it. I also think a border of 5 pixels looks good. The border is white by default.

3) Morph the sequence with `python main.py -morph -images <ALIGN_OUTPUT> -td <TRANSITION_DUR> -pd <PAUSE_DUR> -fps <FPS> -out <OUTPUT_NAME>.mp4`

This will create a video named OUTPUT_NAME in the root directory with the desired parameters. Note that TRANSITION_DUR and PAUSE_DUR are floats while FPS is an integer. 

You may again be prompted to choose the correct face.


Optional: Add music with

`ffmpeg -i <OUTPUT_NAME> -i <AUDIO_TRACK> -map 0:v -map 1:a -c copy -shortest -pix_fmt yuv420p <NEW_OUTPUT_NAME>.mov`

e.g. 
ffmpeg -i demo.mp4 -i /Users/Andy/Music/iTunes/iTunes\ Media/Music/Family\ Of\ The\ Year/Loma\ Vista/05\ Hero.mp3 -map 0:v -map 1:a -c copy -shortest -pix_fmt yuv420p output.mov


To create an average face:
================================

Follow steps 1) - 2). You probably don't want to overlay or use a border though. Then run

`python main.py -average -images <ALIGN_OUTPUT> -out <OUTPUT_NAME>.jpg`


Notes
================================

-If you have photos that are already the same size and all of the faces are more or less aligned already (e.g. roster photos or yearbook photos), you can skip steps 1 and 2 and just use the directory of photos as your aligned images.

-IMPORTANT: Download shape_predictor_68_face_landmarks.dat and place it in the parent directory. 

