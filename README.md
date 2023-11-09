# Deep_Learning_Challenge

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. The goal is to use knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.


From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special consideration for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively


## Instructions 

### STEP 1: PREPROCESS THE DATA: 

Using your knowledge of Pandas and the Scikit-Learn’s StandardScaler(), you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Step 2

Using the information we have provided in the starter code, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
 * What variable(s) are considered the target(s) for your model?
 * What variable(s) are considered the feature(s) for your model?
2. Drop the EIN and NAME columns.
3. Determine the number of unique values for each column.
4. For those columns that have more than 10 unique values, determine the number of data points for each unique value.
5. use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6. Use pd.get_dummies() to encode categorical variables


### STEP 2: COMPILE,TRAIN, EVALUATE THE MODEL: 


Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the jupyter notebook where you’ve already performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every 5 epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.h5.

### STEP 3: OPTIMIZE THE MODEL:

Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%.

1. Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
  * Dropping more or fewer columns.
  * Creating more bins for rare occurrences in columns.
  * Increasing or decreasing the number of values for each bin.
2. Adding more neurons to a hidden layer.
3. Adding more hidden layers.
4. Using different activation functions for the hidden layers.
5. Adding or reducing the number of epochs to the training regimen.


Then: 

1) Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
2) Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.
3) Preprocess the dataset like you did in Step 1, taking into account any modifications to optimize the model.
4) Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
5) Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5.


## My Final Analysis and Report 

**Overview:** The purpose of the model was to create an algorithm to help Alphabet Soup, predict whether or not applicants for funding will be successful. The model was a binary classifier that was able to predict with a fairly high degree of accuracy if the funding would be successful or not.

**Results:** Given Below is my Final Report and Analysis of the Neural Network Model along with answers to the questions posed in the assignment:

> ### Data Preprocessing

1) What variable(s) are the target(s) for your model?
   The variable for the Target was identified as the column `IS_SUCCESSFUL`.
2) What variable(s) are the features for your model?
   The following columns were considered as features for the model:
    > NAME
    
    > APPLICATION_TYPE
    
    > AFFILIATION
    
    > CLASSIFICATION
    
    > USE_CASE
    
    > ORGANIZATION
    
    >  STATUS
    
    > INCOME_AMT
    
    > SPECIAL_CONSIDERATIONS
    
    > ASK_AMT

3) What variable(s) are neither targets nor features and should be removed from the input data?
  The column or variable that can be removed is `EIN` as it is an identifier for the applicant organization and has no impact on the behavior of the model.

> ### Compiling, Training, and Evaluating the Model

1) How many neurons, layers, and activation functions did you select for your neural network model, and why?
   In the Optimized version of the model, I used 3 hidden layers each with multiple neurons which increased the accuracy to <75% to 79%. The Initial model had only 2 layers. In addition, I also increased the number of epochs from 20 to    100 in my model. Increasing the number of hidden layers helped achieve an accuracy score of **79.07%**, compared to the original score of **72.92%**  

2) Were you able to achieve the target model performance?
   **Yes**, I was able to achieve an accuracy score of **79%** when the target was to achieve an accuracy over **75%**
   
3) What steps did you take to try and increase model performance?
   **I took the below steps:**
  > Instead of dropping both the `EIN` and `Name` columns, only the `EIN` column was dropped. However, only the names which appeared more than 5 times were considered.
  > Added a 3rd Activation Layer to the model in the following order to boost the accuracy to > 75%:

   My Layers:
    First Layer `tanh`
    Second Layer `relu`
    Third Layer `relu`
    Output Layer: `sigmoid`

**Summary:**
> Overall, by optimizing the model we are able to increase the accuracy to above 79%.

> This means we are able to correctly classify each of the points in the test data 79% of the time. In other words, an applicant has a close to 80% chance of being successful if they have the following:

  * The NAME of the applicant appears more than 5 times (they have applied more than 5 times)
  * The type of APPLICATION is one of the following: T3, T4, T5, T6 and T19
  * The application has the following values for CLASSIFICATION: C1000, C1200, C2000, C2100 and C3000.

   

