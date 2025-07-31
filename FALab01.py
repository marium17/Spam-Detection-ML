# Step 1: Training data as list of dictionaries
messages = [
    {"text": "send us your password", "label": "spam"},
    {"text": "review our website", "label": "spam"},
    {"text": "send your account", "label": "spam"},
    {"text": "your account is blocked", "label": "spam"},
    {"text": "your account report", "label": "notspam"},
    {"text": "benefits physical activity", "label": "notspam"},
    {"text": "send us important news", "label": "spam"},
    {"text": "benefits of ibadat", "label": "notspam"}
]

# Word counters and counts
spam_words = {}
notspam_words = {}
spam_count = 0
notspam_count = 0

# Vocabulary list 
vocabulary = []

# Loop through the training messages
for item in messages:
    message = item["text"]
    label = item["label"]
    words = message.lower().split()

    for word in words:
        if word not in vocabulary:
            vocabulary.append(word)
    
    if label == "spam":
        spam_count += 1
        for word in words:
            spam_words[word] = spam_words.get(word, 0) + 1
           
    else:
        notspam_count += 1
        for word in words:
            # notspam_words[word] = notspam_words.get(word, 0) + 1
            notspam_words[word] = notspam_words.get(word,0) + 1

# Step 2: Prediction function
def predict(message):
    # Prior probabilities
    total_messages = spam_count + notspam_count
    p_spam = spam_count / total_messages
    p_notspam = notspam_count / total_messages

    # Total word counts and vocabulary size
    total_spam_words = sum(spam_words.values())
    total_notspam_words = sum(notspam_words.values())
    vocab_size = len(vocabulary)

    # Start with prior probabilities
    spam_prob = p_spam
    notspam_prob = p_notspam

    # Multiply likelihoods with add-1 smoothing
    for word in message.lower().split():
        spam_prob *= (spam_words.get(word, 0) + 1) / (total_spam_words + vocab_size)
        notspam_prob *= (notspam_words.get(word, 0) + 1) / (total_notspam_words + vocab_size)

    # Compare final probabilities
    if spam_prob > notspam_prob:
        return "Spam"
    else:
        return "Not Spam"
    
def learn_from_user_feedback(message, actual_label):
    # Same logic as training phase
    words = message.lower().split()
    global spam_count, notspam_count

    if actual_label == "spam":
        spam_count += 1
        for word in words:
            spam_words[word] = spam_words.get(word, 0) + 1
    else:
        notspam_count += 1
        for word in words:
            notspam_words[word] = notspam_words.get(word, 0) + 1

# Step 3: Test the classifier
msg = input("Enter a message: ")
print("This message is:", predict(msg))
# Ask user if it was correct
correct = input("Is this correct? (yes/no): ")
if correct.lower() == "no":
    actual = input("What is the correct label? (spam/notspam): ")
    learn_from_user_feedback(msg, actual)
    
print(spam_words)
print(notspam_words)
