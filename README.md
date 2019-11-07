# reddit_word_2_vec
Applying Word 2 Vec Algorithm on Reddit comments

My take on word 2 vec algorithm implementation from Medium [post](https://medium.com/ml2vec/using-word2vec-to-analyze-reddit-comments-28945d8cee57). 

Stumbled upon the post while trying to implement Word 2 Vec algorithm for [Reddit](reddit.com) comments. Implementation is almost same. Highly advised to read the blog post and then come back to this. 

All the credit for original implementation goes to https://github.com/ravishchawla/word_2_vec.

Dataset can be downloaded from Kaggle [here](https://www.kaggle.com/reddit/reddit-comments-may-2015). After downloading the dataset, create a new folder called "dataset" in directory of project and extract the downloaded dataset into this folder.

## What is different ?
Mainly wrote this code in midst of learning process. Original excercise is completely in IPython notebook while this is to be executed as a script.

Since dataset is very large (~30 GB) and original excercise was done on AWS P4.2xLarge instance, with 60 GB RAM, some changes were made to make this run of normal PC's albeit with variable(preferably lesser) number of comments.

Code refactor and flask based web interface for changing various parameters and observe effect on ouput. Screenshots can be seen below.

Front end templates taken from [Colorlib](https://colorlib.com/).

https://github.com/zz-xx/reddit_word_2_vec/blob/master/imgs/home.PNG

https://github.com/zz-xx/reddit_word_2_vec/blob/master/imgs/out.PNG
