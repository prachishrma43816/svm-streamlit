import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Title
# -----------------------------
st.title("📧 Email Spam Classifier (SVM)")
st.write("""
This app classifies emails as **Spam** or **Not Spam** using a Linear SVM model.
""")

# -----------------------------
# Step 1: Dataset
# -----------------------------
emails = [
    "Congratulations! You’ve won a free iPhone",
    "Claim your lottery prize now",
    "Exclusive deal just for you",
    "Act fast! Limited-time offer",
    "Click here to secure your reward",
    "Win cash prizes instantly by signing up",
    "Limited-time discount on luxury watches",
    "Get rich quick with this secret method",
    "Hello, how are you today",
    "Please find the attached report",
    "Thank you for your support",
    "The project deadline is next week",
    "Can we reschedule the meeting to tomorrow",
    "Your invoice for last month is attached",
    "Looking forward to our call later today",
    "Don’t forget the team lunch tomorrow",
    "Meeting agenda has been updated",
    "Here are the notes from yesterday’s discussion",
    "Please confirm your attendance for the workshop",
    "Let’s finalize the budget proposal by Friday"
]

labels = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]

# -----------------------------
# Step 2: Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2),
    max_df=0.9,
    min_df=1
)

X = vectorizer.fit_transform(emails)

# -----------------------------
# Step 3: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# -----------------------------
# Step 4: Train SVM Model
# -----------------------------
svm_model = LinearSVC(C=1.0)
svm_model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader(f"📊 Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# Step 5: User Input for Prediction
# -----------------------------
st.sidebar.header("Enter an email message to classify")
user_email = st.sidebar.text_area("Email Content", "")

if st.sidebar.button("Predict"):
    if user_email.strip() == "":
        st.warning("Please enter an email message to classify.")
    else:
        user_vectorized = vectorizer.transform([user_email])
        prediction = svm_model.predict(user_vectorized)[0]
        if prediction == 1:
            st.error("🚨 Result: The email is **SPAM**.")
        else:
            st.success("✅ Result: The email is **NOT SPAM**.")
