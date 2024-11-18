import pandas as pd
import numpy as np
import string
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка датасета
data_path = 'SMSSpamCollection'
df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])


# Предобработка текста
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text.split()


df['clean_message'] = df['message'].apply(preprocess_text)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df['clean_message'], df['label'], test_size=0.2, random_state=42)


# Функция для подсчета вероятностей
class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_doc_counts = defaultdict(int)
        self.class_totals = defaultdict(int)
        self.vocab = set()

    def fit(self, X, y):
        self.classes = set(y)
        for text, label in zip(X, y):
            self.class_doc_counts[label] += 1
            for word in text:
                self.class_word_counts[label][word] += 1
                self.class_totals[label] += 1
                self.vocab.add(word)

    def predict(self, X):
        predictions = []
        for text in X:
            class_probs = {}
            for c in self.classes:
                # Вычисляем P(C)
                class_prob = np.log(self.class_doc_counts[c] / sum(self.class_doc_counts.values()))

                # Вычисляем P(X | C)
                word_probs = 0
                for word in text:
                    word_count = self.class_word_counts[c].get(word, 0) + 1
                    word_prob = np.log(word_count / (self.class_totals[c] + len(self.vocab)))
                    word_probs += word_prob
                class_probs[c] = class_prob + word_probs
            # Выбираем класс с максимальной вероятностью
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean([pred == true for pred, true in zip(predictions, y)])


# Обучение и тестирование модели
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# Оценка на тестовой выборке
accuracy = nb_classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

y_pred = nb_classifier.predict(X_test)


# Распределение сообщений по классам
plt.figure(figsize=(8, 5))
sns.countplot(x=df['label'])
plt.title('Распределение по классам')
plt.xlabel('Класс')
plt.ylabel('Количество сообщений')
plt.show()

# Топ-10 самых частых слов в спам и не спам сообщениях
def get_top_words(label, n=10):
    words = []
    for text, lbl in zip(df['clean_message'], df['label']):
        if lbl == label:
            words.extend(text)
    return Counter(words).most_common(n)

spam_top_words = get_top_words('spam')
ham_top_words = get_top_words('ham')

# Построение графиков для спам и не спам сообщений
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
spam_words, spam_counts = zip(*spam_top_words)
ham_words, ham_counts = zip(*ham_top_words)

ax[0].barh(spam_words, spam_counts, color='red')
ax[0].set_title('Топ-10 слов в спам-сообщениях')
ax[0].invert_yaxis()

ax[1].barh(ham_words, ham_counts, color='green')
ax[1].set_title('Топ-10 слов в не спам-сообщениях')
ax[1].invert_yaxis()

plt.show()

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred, labels=['spam', 'ham'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['spam', 'ham'])
disp.plot(cmap='Blues')
plt.title('Матрица ошибок')
plt.show()
