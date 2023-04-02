# Quiz Bowl Question Classifier

The purpose of this project is to correctly classify Quiz Bowl questions into their respective categories and subcategories using a machine learning approach. The main techniques used are tf-idf vectorization, naive bayes classification, and the cosine similarity metric. 

## Motivation
Being able to correctly classify questions is a nontrivial task that would help improve the quality of existing question databases as well as automate the assignment of new questions as they are added. This project was originally created during the final portion of my Applied Data Science (DS 325) class. I have since continued to add featues, including better model performances and front-end functionality where users can interact with the existing models. 

## Repository Structure

The general structure of this repository contains three main directories, as shown below:

📦Quiz Bowl Question Classifier
    ┣ 📂Jupyter Notebook
    ┃  ┣ 📂Models
    ┃  ┣ 📂Project_Data
    ┣ 📂Flask_App
    ┗ 📂Documentation

**Jupter Notebook**: Contains a completely executed jupyter notebook containing the step-by-step machine learning process. This also has folders containing all input data and the generated models.

**Flask_App**: Contains the flask application, where users can test the models with new questions.

**Documentation**: Contains a paper, presentation slides, and presentation video discussing the results of this project.

## General Project Pipeline 
The below list shows the general methodology / data pipeline for this project:

1. Data Collection
    - Data gathered from [quizdb.org](https://www.quizdb.org/) (now defunct) and [quizbowlpackets.com](https://quizbowlpackets.com/)
2. Preprocessing
    - Tokenize, lowercase, lemmatize, and remove stopwords
3. Visualization
    - Used the t-SNE algorithm
4. TF-IDF Vectorization
5. Classification
    - Method One: Naive Bayes
    - Method Two: Cosine Similarity
6. Scoring / Analysis
7. Deployment


## How to use this project
Please feel to reference, use, and/or adapt any or all parts of this repository, including code, methodology or analysis in any way you wish. I encourage everyone to try and improve the performance of my current models! The data in this repository is not owned by me, and as such is subject to any restictions put forth by the respective owners. 

## Credits:
* [quizdb.org](https://www.quizdb.org/) for inspiration and training data. (This has since been shut-down, but is based on data from [quizbowlpackets.com](https://quizbowlpackets.com/))
* [triviabliss.com](https://triviabliss.com/) for additional comparison data
* Professor James Puckett for instruction and guidance