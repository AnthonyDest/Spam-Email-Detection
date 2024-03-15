import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import matplotlib.pyplot as plt


def getdata(name): 
    data = pd.read_csv(name)
    return data 

def preprocessing(data): 
    # extracting features and labels
    X = data['text']  # Features - the email text
    y = data['label_num']  # Target variable - spam or not spam
    
    # TF-IDF vectorization of the email texts
    #vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    vectorizer = CountVectorizer(stop_words='english', max_features=3000)
    x_vectorized = vectorizer.fit_transform(X)

    return x_vectorized, y, vectorizer

def ml(x_vectorized, y, n_estimators): 
    # splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x_vectorized, y, test_size=0.2, random_state=42)
    
    # Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    #n_estimators = number of trees. more trees, generally more stability, but also more computer power 
    #random_state = the randomness of the bootstrapping of the samples when building trees. doesn't impact performance. keep it at a number just makes sure the results are reproducible. 
    rf_classifier.fit(X_train, y_train)
    
    # predictions on the test set
    predictions = rf_classifier.predict(X_test) #1 is spam, 0 is not spam 
    #print (predictions)
    
    # evaluate the classifier
    accuracy = accuracy_score(y_test, predictions)
    classification_report_text = classification_report(y_test, predictions)
    
    #print (classification_report_text)
    joblib.dump(rf_classifier, 'email_spam_model.pkl')
    return accuracy 
    

def test(vectorizer, email): 
    loaded_model = joblib.load('email_spam_model.pkl')
    new_email = [email]
    new_email = vectorizer.transform(new_email)  
    prediction = loaded_model.predict(new_email)
    
    if prediction[0]: #is spam 
        return True
    return False #not spam 

data = getdata('spam_dataset.csv')
x_vectorized, y, vectorizer = preprocessing(data)
accuracies = []
for i in range (1, 130): 
    a = ml(x_vectorized, y, i)
    print (i, ': ', a) 
    accuracies.append(a)

plt.plot(accuracies)

result1 = test (vectorizer, "Thank you for providing a signed Customer Services Agreement. A $250 security deposit will be applied to your first bill. If you choose to setup the Preauthorized Payment Plan, the deposit will be waived. If you can provide an up-to-date Electricity or Gas Reference Letter (within last 2 years) from a previous supplier, showing at least 12 months of Good Payment History, we can also waive the deposit.")
print ("is spam: ", result1)
result2 = test (vectorizer, "Congratulations! You've won a prize. Claim it now.")
print ("is spam: ", result2)