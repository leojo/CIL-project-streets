# CIL-project-streets

REQUIREMENTS

Python 2.7, preferably with Virtualenv.

SET-UP

In your virtual environment, install all dependencies for the project by running

> pip install -r requirements.txt

PREPROCESSING

To smooth all images (the training and test images) in one step, we prepared a Matlab script. The code for applying the Fast Local Laplacian is taken from {insert url} and modified to smooth all images in a directory with the same parameters.

1. Open matlab and select the folder "fast_local_laplacian_filters" as the working directory (or add it to the path)
2. Open the file "smooth_all.m"
3. Adjust the paths on line 4 to 7
  + path_orig: Relative path to the original training images
  + path_new: Relative path where the new smoothed images are saved
  + path_test_orig: Relative path to the directory which contains the test-directories
  + path_test_new: Relative path where the test-directories containing the smoothed test-images are saved
4. Run the script (this lasts for about 5 minutes)
5. You can find the smoothed images in the new directories. With the current implementation, it is only possible to store the smoothed images as jpegs. Since we are reading in our python-scripts the content of the directories and looping through each file, this was not a limitation or problem for further progressing.

TRAINING

After performing the preprocessing step above, run the command

> python my_cnn.py \[prediction_dir\] \[train_data_dir\] \[train_filetype\] \[test_data_dir\] \[test_filetype\] \[patch_size\] \[num_patches\] \[num_epochs\] \[minibatch_size\] \[Resume_from_save\]

If you don't want to utilize the configurations provided, run this command: 
> python my_cnn.py submission_masks ../data/training_smooth jpg ../data/test_set_smooth jpg 64 100 20 5 False

RESUME A FAILED TRAINING

Run the command above with Resume_from_save = True. The script will detect the last checkpoint and continue training.

TESTING

Whether you decided to use a pre-trained model (stored in the root of the cnn folder) or to train a model of your own saved in the same place, now run

> python my_cnn.py \[prediction_dir\] \[train_data_dir\] \[train_filetype\] \[test_data_dir\] \[test_filetype\] \[patch_size\] \[num_patches\] \[num_epochs\] \[minibatch_size\] \[Resume_from_save\]

where Resume_from_save = True. It is necessary that the model completed training for this to work, otherwise training will resume as explained above.

POST-PROCESSING

Now go to the folder watershed/ and run

> python watershed_postproc.py

CREATING A SUBMISSION FILE

You need to run

> python mask_to_submission.py 

to create the .csv submission file. This will create a submission from the output masks stored in watershed/submission_masks_watershed/.

OTHER

To create a plot that demonstrates the difference in loss over time for the baseline model vs our model, run

> python generate_plots.py

Make sure you have run both our model and the baseline model beforehand to generate datapoints.

The baseline.py is the baseline as provided by the instructors (only slightly modified for our directory structure and for output plots) and is included here for completeness.
