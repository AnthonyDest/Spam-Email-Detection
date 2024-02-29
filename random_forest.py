import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def test(email): 
    loaded_model = joblib.load('email_spam_model.pkl')
    new_email = [email]
    new_email = tfidf_vectorizer.transform(new_email)  
    prediction = loaded_model.predict(new_email)
    
    if prediction[0]: #is spam 
        return True
    return False #not spam 


data = pd.read_csv('spam_dataset.csv')

# extracting features and labels
X = data['text']  # Features - the email text
y = data['label_num']  # Target variable - spam or not spam

# TF-IDF vectorization of the email texts
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# predictions on the test set
predictions = rf_classifier.predict(X_test) #1 is spam, 0 is not spam 
print (predictions)

# evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
classification_report_text = classification_report(y_test, predictions)

print (accuracy, classification_report_text) 
joblib.dump(rf_classifier, 'email_spam_model.pkl')

result1 = test ("Thank you for providing a signed Customer Services Agreement. A $250 security deposit will be applied to your first bill. If you choose to setup the Preauthorized Payment Plan, the deposit will be waived. If you can provide an up-to-date Electricity or Gas Reference Letter (within last 2 years) from a previous supplier, showing at least 12 months of Good Payment History, we can also waive the deposit.")
print ("is spam: ", result1)
result2 = test ("Congratulations! You've won a prize. Claim it now.")
print ("is spam: ", result2)

