# fast-rcnn-train-another-dataset

If you think those can help you, please give me star~~~~thank you so much :)
如果你觉得我这些能对你有帮助，请给我一个星星，原谅我那么不要脸~~ :） 

Those three file can help you to train fast-rcnn on your own dataset.
You can follow the steps in my blog http://www.cnblogs.com/louyihang-loves-baiyan/p/4903231.html 
In the blog, I've list the detials and something need to be take care when modifying and training the model.

kakou.py is similar to the pascal.py which is used to read your own dataset.
factory.py is a factory class used to generate imdb class that is the base class of kakou and pascal.
carfacetest.py is the test interface that I wrote based on the demo.py, and the detection data will output to the txt file
in the current directory.

You should place kakou.py and factory.py in the fast-rcnn/lib/dataset, and place the carfacetest.py in the fast-rcnn/tools
