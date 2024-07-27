Covid-19 X-ray classifier

This is a neural network model created to classify X-ray pictures if the patient  has Covid, viral pneumonia or if the individual is of normal health.

The dataset is from Kaggle, link: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset?resource=download

The dataset contains 378 images in total. The dataset is divided to train and test files, and these files both contain three files named Covid, Normal and Viral Pneumonia. The training data contains 312 images and the test data 66 images.

The classifier is trained by using Python and Tensorflow in Visual Studio Code. Images are rezised to 150x150 pixels, since the images are originally of different sizes. 

The code is created by prompting ChatGPT. 

This directly doesn't per se answer a humanities/social science research question, but since I study cognitive science, neural networks are a central part of our field. I learned that by prompting ChatGPT I can create neural networks pretty easily, and I can continue creating more with it. This type of neural networks are useful probably in all sciences to notice patterns that might not be detectable to humans. Maybe a closer question for cognitive science would have been to work with patterns of the brain, but I think this is a good starting point for learning to use neural networks. These sorts of pattern recognition networks are really useful for medicine, since specific patterns from multidimensional data can help detect cancer and other illnesses even earlier than what has been possible.

The best test accuracy that I came to was 0.8787878751754761, meaning ~88%. I did run the script a couple of times and the accuracy wasn't stable, meaning that using this same script, different level of accuracy can be reached. Lower numbers were around 60% accuracy. One of the ways to make the results more reliable is adding more data, meaning more images to try and generalize features from.

The code was modified during the process to include more data transformations, like random rotations or brightness changes in the 0.2 range, meaning the maximum of 20 percent. A pretrained model was added as well, and the training was added onto that.

