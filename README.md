# Threat Detection
There have been many mass shootings over the year, and each year it has been increasing. Many people lose their lives because the threat wasn't neutralized early enough. Cameras rely on human eyes to detect such actions and a human's eye/reaction time is quite frankly just not good enough.

We can lower casualties in mass shootings earlier by having a computer do the work for us. What if a computer is able to detect a threat before it begans?

This work helps approach the problem to find a solution.

![Demo](https://media.giphy.com/media/VcxJRTfa2A0UKthWEx/giphy.gif)

## Summary Contents

- [Image Classification](https://github.com/jeffersonzaki/Threat-Detection/tree/master/Image-Classification) - Contains all of the data and notebooks that were wokred on.

  - [Data](https://github.com/jeffersonzaki/Threat-Detection/tree/master/Image-Classification/Data-Images) - Contains two folders, one of [assault rifles](https://github.com/jeffersonzaki/Threat-Detection/tree/master/Image-Classification/Data-Images/Assault%20Rifle) and the other of [handguns](https://github.com/jeffersonzaki/Threat-Detection/tree/master/Image-Classification/Data-Images/Handgun)

  - [Python Script](https://github.com/jeffersonzaki/Threat-Detection/tree/master/Image-Classification/Script) - Contains a script that accesss the contents of a bucket. This script was then connected to crontab in order to continue to refresh the AWS page, because of an 2 hour access time.

  - [Augmented Convolutional Neural Network](https://github.com/jeffersonzaki/Threat-Detection/blob/master/Image-Classification/augmented_cnn.ipynb) - Contains a CNN that was ran with augmented data in an attempt to imrove an already existing accuarcy score.

  - [Specified Convolutional Neural Network](https://github.com/jeffersonzaki/Threat-Detection/blob/master/Image-Classification/augmented_cnn.ipynb) - Contains the main train, test, and validation scores from the original CNN models.

  - [Saved Entire Model](https://github.com/jeffersonzaki/Threat-Detection/blob/master/Image-Classification/specified_model.hdf5) - A file that contains the saved model that performed the best.

  - [Saved Model Weights](https://github.com/jeffersonzaki/Threat-Detection/blob/master/Image-Classification/specified_weights.hdf5) - A file of the weights from the model that performed the best.

  - [Web Driver](https://github.com/jeffersonzaki/Threat-Detection/blob/master/Image-Classification/web_driver.ipynb) - Used to scrape automatically scrape images from the internet using chromedriver and selenium.
  
- [Gitignore](https://github.com/jeffersonzaki/Threat-Detection/blob/master/.gitignore) - Contains files that were ignored to github.
  
- [Business Plan](https://github.com/jeffersonzaki/Threat-Detection/blob/master/Business%20Plan.pdf) - Contains a pdf of the business plan to put this idea into action.
 
- [ReadME](https://github.com/jeffersonzaki/Threat-Detection/blob/master/README.md) - Shows a summary of the repo.

## Project Member
[Zaki Jefferson](https://github.com/jeffersonzaki)

## Project Scenario
AIVision is a start up company that's looking to make a mark by developing threat detection technology. Their business model is to retrieve a licensing deal with another company and continue to add value to their IP to make it better as the years progress.

## Responsibilities
[Zaki Jefferson](https://github.com/jeffersonzaki)

- Retrieve image data
- Develop a variety of Convolutional Neural Networks
- Augment data
- Develp web app
- Clean notebook
- Develop presentation

ALL DONE ON AWS

## Data
[Images](https://github.com/jeffersonzaki/Threat-Detection/tree/master/Image-Classification/Data-Images) were scrapped from the internet using selenium
