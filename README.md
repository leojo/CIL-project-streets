# CIL-project-streets

Requirements

Python 2.7, preferably with Virtualenv.

Set-up

In your virtual environment, install all dependencies for the project by running

> pip install -r requirements.txt

Testing

Perform the preprocessing step above, then run

python my_cnn.py

Preprocessing




How to smooth the images? 

To smooth all images (the training and test images) in one step, we prepared a Matlab script. The code for applying the Fast Local Laplacian is taken from {insert url} and modified to smooth all images in a directory with the same parameters.

* Open matlab and select the folder "fast\_local\_laplacian\_filters" as the working directory (or add it to the path)
* Open the file "smooth\_all.m"
* Adjust the paths on line 4 to 7
 + path\_orig: Relative path to the original training images
 + path\_new: Relative path where the new smoothed images are saved
 + path\_test\_orig: Relative path to the directory which contains the test-directories
 + path\_test\_new: Relative path where the test-directories containing the smoothed test-images are saved
* Run the script (this lasts for about 5 minutes)
* You can find the smoothed images in the new directories. With the current implementation, it is only possible to store the smoothed images as jpegs. Since we are reading in our python-scripts the content of the directories and looping through each file, this was not a limitation or problem for further progressing.
