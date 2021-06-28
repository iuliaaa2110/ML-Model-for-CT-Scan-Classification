# ML-Model-for-CT-Scan-Classification

Contest Link : https://www.kaggle.com/c/ai-unibuc-23-31-2021/overview
<br />*Score:* 0.76581% Accuracy
<br />*Entries:* 8




<br />*Overview:*
This is a CT scan classification challenge in which competitors have to train machine learning models on a data set containing computer tomography (CT) scans of lungs. Competitors are scored based on the classification accuracy on a given test set. For each test image, the participants have to predict its class label.




<br />*Task:*
Participants have to train a model for CT scan classification. This is a multi-way classification task in which an image must be classified into one of the three classes (native, arterial, venous).

The training data is composed of 15,000 image files. The validation set is composed of 4,500 image files. The test is composed of 3,900 image files.

<br />*File Descriptions:*
train.txt - the training metadata file containing the training image file names and the corresponding labels (one example per row)
validation.txt - the validation metadata file containing the validation image file names and the corresponding labels (one example per row)
test.txt - the test metadata file containing the test image file names (one sample per row)
sample_submission.txt - a sample submission file in the correct format
Data Format
Metadata Files
The metadata files are provided in the following format based on comma separated values:

000001.png,0
000002.png,1
...
Each line represents an example where:

The first column shows the image file name of the example.
The second column is the label associated to the example.
Image Files
The image files are provided in .png format.





Base Rules
One account per participant
You cannot sign up to Kaggle from multiple accounts and therefore you cannot submit from multiple accounts.

You must use your full name (familiy name + first name) for the leaderboard.

Failure to comply will result in disqualification.

No private sharing outside teams
Privately or publicly sharing code or data outside of teams is not permitted.

Copy-pasting code from the web is not permitted (no exceptions allowed, not even if you indicate the source).

Failure to comply will result in disqualification.

Team Mergers
Team mergers are not allowed in this competition.

Team Limits
The maximum size of a team is 1 participant.

Submission Limits
You may submit a maximum of 2 entries per day.

You may select up to 2 final submissions for judging.

Competition Timeline
Start Date: 18.04.2021 11:00 AM EEST

Merger Deadline: None

Entry Deadline: None

End Date: 19.05.2021 11:59 PM UTC

Grading
Your score in this competition will reflect your final grade for the project. Your grade will be greater than 5 only if you manage to beat the baseline submission. The grades will be given based on the final ranking as follows:

The top 30 competitors can receive a full grade (10 out of 10)
The following 30 competitors (ranks 31 to 60) can receive a grade up to 9
The following 30 competitors (ranks 61 to 90) can receive a grade up to 8
The following 20 competitors (ranks 91 to 110) can receive a grade up to 7
The following 20 competitors (ranks 111 to 130) can receive a grade up to 6
All other competitors will receive the grade 5 (given that they manage to beat the baseline)
Project Documentation
Your grade will be given only if you provide the code and a documentation for the proposed approaches. The documentation should include:

The description of your machine learning approaches including the tested models (k-NN, SVM, MLP, etc.). Details should also include feature extraction (if present) and hyperparameter choices (learning rate, performance function, regularization, etc.). A minimum of two pages (excluding tables and figures) is expected. Documenting all the tried approaches (even unsuccessful ones) is important.
Classification accuracy rates and confusion matrices for the provided validation set.
The python code of your model should include explanatory comments.
The *.py files must be placed in the same folder (without organization into subfolders) with the documentation. The folder should be named '{family_name}_{first_name}_{group}'.
Competitors must convert *.ipynb files to *.py (failure to comply will result in disqualification).
The documentation and the *.py files must be archived in a single *.zip file.
Important notes regarding the submitted code and documentation:

The deadline for submitting the documentation is 21.05.2021 23:59 EEST (2 days after competition ends).
Code and documentation is subject to plagiarism verification and can lead to disqualification.
The project documentation must be submitted to your lab supervisor by e-mail and presented during week 14, after the competition ends.
