from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Dataset (Text messages and labels)
emails = [
    "Win free money now", 
    "Hello friend how are you", 
    "Get cheap loans today", 
    "Meeting tomorrow at 10 am"
]
labels = [1, 0, 1, 0] # 1 = Spam, 0 = Not Spam

# 2. Convert Text to Numerical Features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# 3. Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X, labels)

# 4. Test New Email
test_email = ["Win a free prize today"]
test_X = vectorizer.transform(test_email)
prediction = model.predict(test_X)

if prediction[0] == 1:
    print("Result: SPAM Email")
else:
    print("Result: NOT SPAM Email")
