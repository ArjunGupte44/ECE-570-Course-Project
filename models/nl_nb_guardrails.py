import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


SAFE_QUESTIONS_PATH = r"C:\Users\agupt\OneDrive\Documents\Purdue\Junior Year\Fall Semester\ECE 570\Course Project\ECE_570_Course_Project\data\user_questions\safe_questions.txt"
POTENTIAL_QUESTIONS_PATH = r"C:\Users\agupt\OneDrive\Documents\Purdue\Junior Year\Fall Semester\ECE 570\Course Project\ECE_570_Course_Project\data\user_questions\potential_questions.txt"
VIOLATIONS_QUESTIONS_PATH = r"C:\Users\agupt\OneDrive\Documents\Purdue\Junior Year\Fall Semester\ECE 570\Course Project\ECE_570_Course_Project\data\user_questions\violations_questions.txt"

# Step 1: Load the dataset
data = []
labels = []

with open(SAFE_QUESTIONS_PATH, "r") as file:
    for line in file:
        # if len(data) < 50:
        data.append(line.strip())
        labels.append(0)

# print(data)
with open(VIOLATIONS_QUESTIONS_PATH, "r") as file:
    for line in file:
        # if len(data) < 100:
        data.append(line.strip())
        labels.append(1)

print(len(labels))
# Step 3: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
print(X_train)
print(y_train)
# Step 4: Convert the text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 1: Initialize the SVM model
nb_model = MultinomialNB(alpha=100)

# Step 2: Train the model
nb_model.fit(X_train_tfidf, y_train)

# Step 3: Predict on the test data
y_pred_svm = nb_model.predict(X_test_tfidf)

# Step 4: Evaluate the SVM model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report for SVM:\n", classification_report(y_test, y_pred_svm))

scores = cross_val_score(nb_model, X_train_tfidf, y_train, cv=5)  # 5-fold cross-validation
print("Cross-validation accuracy:", scores.mean())


# Create the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_svm)
classes = ["Safe", "Direct Violation"]

# Visualize the confusion matrix using Seaborn
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='viridis', xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
