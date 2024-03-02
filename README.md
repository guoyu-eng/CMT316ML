# CMT316ML
## Overview

There are Two part of this project.The first part uses models of classification and regression to predict house prices, the second part deals with, preprocessing, dimensionality reduction, feature selection, develop set application, overfitting, and evaluation and discussion of model performance, with an additional accompanying comparison of the models (Random Forest), and a comparison and discussion of the performance of the random forest model and the vector machine model in this project.



## Part1
### Notification
Part1 aims to train machine learning models to predict the house price of a unit area using the "real_state" dataset. The dataset contains various house properties, and the goal is to develop both regression and classification models to predict house prices and classify houses as "expensive" or "not-expensive" based on their unit area price.

### Classification Model:
This model is designed with two main running files, training file and test file, but with modification, the performance of training file and test file is measured by Root Mean Square Error (RMSE). This model designs two main running files, training file and test file, but is modified so that the training file can perform all tasks at the same time .( training and testing)
<pre>
python regression_train
</pre>
Export the model while completing model training, or run a separate test file to evaluate the model (make sure you have the trained model in a unified directory)

<pre>
python regression_test
</pre>


### Regression Model:
same upon
<pre>
python assignment1
</pre>
Export the model while completing model training, or run a separate test file to evaluate the model (make sure you have the trained model in a unified directory)

<pre>
python assi1test
</pre>





## Part2
### Notification
By the way, the en_core_web_sm model works fine on my Mac (M1) and Dell windows, but it won't download on my VM, so if it doesn't work, comment out the first function called ".preProcessing" and use the second one ( it under the first one, and you should find the note reminder) . it  should be found in every file! and while there will be a performance drop, it won't have much of an impact on this project (about 4% drop, still over 80%). Runtime for model training and testing should be less than 200 seconds


### Installation

<pre>
pip install -U spacy
pip install scikit-learn
pip install nltk
pip install chardet
python -m spacy download en_core_web_sm ： download
python -m spacy download en_core_web_sm
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0.tar.gz
pip install --upgrade spacy pydantic
</pre>

### How to use

Same upon, the model will be output, and I set the separate test files to facilitate subsequent adjustments and development,Running the following commands will directly complete the training of the model, output, and evaluation of the results,

<pre>
python assignmentpart2.py
</pre>

Of course if you want to save time you can also run my trained model directly to see the evaluation results.(Ensure that the models are already in the file, initially they were)

<pre>
python part_two_test.py
</pre>
### Extra
This will be used to run the Random Forest model (which will complete model training and model testing without model output).

<pre>
python random_forest.py
</pre>

## License

This program is released under the MIT License.

Feel free to experiment with the code and adapt it to your needs. If you find any issues or have suggestions, please open an issue or contribute to the project.



