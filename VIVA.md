# VIVA Preparation - Student Placement Prediction

Short, simple, easy-to-memorize answers for a college viva.

---

**1. What is Machine Learning?**
Machine Learning is a branch of AI where a computer learns patterns from
data and makes predictions or decisions without being explicitly
programmed with fixed rules.

**2. What is supervised learning?**
Supervised learning is when the model is trained on data that already
has the correct answers (labels), so it learns to map inputs to known
outputs. Here, each student record already has a "Placement" label.

**3. Why is this a classification problem?**
Because the output is a category (PLACED or NOT PLACED), not a
continuous number. Predicting categories is called classification;
predicting numbers is called regression.

**4. Why did you choose Logistic Regression?**
It's a simple, fast, well-understood baseline model for binary
classification that also naturally outputs a probability, which we
needed for the "placement probability" feature.

**5. Why did you choose Decision Tree?**
It's easy to visualize and explain (it splits data using simple
if/else-style rules on features like CGPA), which suits a beginner-level
academic project.

**6. Why did you choose Random Forest?**
It combines many decision trees and averages their results, which
usually improves accuracy and reduces overfitting compared to a single
tree. It also lets us calculate feature importance.

**7. Why did you choose KNN?**
KNN is a simple, intuitive algorithm - it classifies a student based on
the most similar students in the training data. It's a good contrast to
the other three "model-based" algorithms since it's instance-based.

**8. What is train-test split?**
It means dividing the dataset into two parts: one to train the model
(train set) and one to check how well it performs on unseen data (test
set). We used an 80/20 split.

**9. Why do we split the dataset?**
To fairly evaluate the model on data it has never seen, so we can tell
if it actually learned useful patterns instead of just memorizing the
training data (overfitting).

**10. What is feature scaling?**
Feature scaling transforms numeric features so they're on a similar
scale (e.g. mean 0, standard deviation 1). We used `StandardScaler`.

**11. Why is scaling required for KNN?**
KNN uses distance between data points to find "nearest neighbors." If
features have very different scales (e.g. CGPA 0-10 vs Coding Score
0-100), the larger-scale feature would dominate the distance
calculation unless everything is scaled first.

**12. What is accuracy?**
The percentage of total predictions that were correct:
(Correct Predictions) / (Total Predictions).

**13. What is precision?**
Of all students the model predicted as "Placed," precision is the
percentage that were actually placed: TP / (TP + FP).

**14. What is recall?**
Of all students who were actually "Placed," recall is the percentage the
model correctly identified: TP / (TP + FN).

**15. What is F1-score?**
The harmonic mean of Precision and Recall - a single balanced score that
is high only when both precision and recall are reasonably good.

**16. What is a confusion matrix?**
A table that shows how many predictions fell into each of four
categories: True Positive, True Negative, False Positive, and False
Negative, compared against the actual outcome.

**17. What are TP, TN, FP and FN?**
- TP (True Positive): predicted Placed, actually Placed
- TN (True Negative): predicted Not Placed, actually Not Placed
- FP (False Positive): predicted Placed, actually Not Placed
- FN (False Negative): predicted Not Placed, actually Placed

**18. What is overfitting?**
When a model learns the training data too well - including its noise -
so it performs great on training data but poorly on new, unseen data.

**19. What is underfitting?**
When a model is too simple to capture the real patterns in the data, so
it performs poorly on both training and test data.

**20. What is feature importance?**
A score (from tree-based models like Random Forest) showing how much
each feature contributed to the model's predictions. Higher importance
means the model relied on that feature more.

**21. Why did you use synthetic data?**
Because real student placement data is private/confidential and hard to
access for a college project. Synthetic data lets us build and
demonstrate the full ML pipeline safely and ethically.

**22. Why can this model not guarantee placement?**
Because real placement depends on many unpredictable real-world factors
(company decisions, market conditions, interview day performance) that
aren't captured in this dataset. The model only finds statistical
patterns in synthetic training data.

**23. How does the prediction probability work?**
Trained classification models can output `predict_proba()`, which gives
the estimated probability of each class based on what the model learned.
We use the probability of the "Placed" class directly - it is never
randomly generated.

**24. How is the best model selected?**
Automatically, by comparing all 4 trained models' F1-scores on the test
set and picking whichever model scored highest. No model is hard-coded
as the "winner."

**25. What is the role of Random Forest?**
It acts as one of the four candidate classifiers, and it is also used
specifically to calculate feature importance, since tree-based models
naturally expose this information.

**26. What is the role of SQLite?**
SQLite is a lightweight, file-based database used here to permanently
store every prediction made (Student ID, inputs, result, probability,
timestamp) so users can view their prediction history later.

**27. Why did you use Tkinter?**
Tkinter is Python's built-in GUI toolkit - no extra installation is
needed, and it's simple enough to build a clean desktop interface for a
beginner-level project.

**28. What preprocessing was performed?**
Missing value checking, duplicate checking, separating features from the
target column, train/test splitting, and feature scaling with
`StandardScaler`.

**29. What are the limitations of the project?**
The dataset is synthetic (not real), the model can't account for
real-world hiring variability, and feature importance shows correlation
learned by the model, not proven causation.

**30. What improvements can be made in the future?**
Using real anonymized data, adding more features, tuning hyperparameters,
using a larger dataset, and deploying the app as a web application.
