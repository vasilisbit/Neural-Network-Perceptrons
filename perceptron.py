import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mplcursors

# Φόρτωση των δεδομένων εκπαίδευσης
train_file_path = 'training_data.csv'
data = pd.read_csv(train_file_path)

# Φόρτωση των δεδομένων testing
test_file_path = 'test_data.csv'
test_data = pd.read_csv(test_file_path)

# Υποθέτουμε ότι τα δεδομένα έχουν τις στήλες 'x1', 'x2' (χαρακτηριστικά) και 'label'
X_train = data[['x1', 'x2']].values
y_train = data['label'].values

X_test = test_data[['x1', 'x2']].values
y_test = test_data['label'].values

# Μετατροπή labels από {0, 1} σε {-1, 1} αν χρειάζεται
y_train = np.where(y_train == 0, -1, y_train)
y_test = np.where(y_test == 0, -1, y_test)

# Κανονικοποίηση των δεδομένων
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Υλοποίηση Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors = []

        for epoch in range(self.epochs):
            error_count = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                if y_predicted != y[idx]:
                    update = self.learning_rate * y[idx]
                    self.weights += update * x_i
                    self.bias += update
                    error_count += 1

            self.errors.append(error_count)

            # Εκτύπωση πληροφοριών για την τρέχουσα εποχή
            print(f"Εποχή {epoch+1}/{self.epochs} - Λάθη: {error_count}")

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

# Εκπαίδευση του Perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=20)
perceptron.fit(X_train, y_train)

# Αξιολόγηση στο test set
y_pred = perceptron.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Ακρίβεια Τεστ: {accuracy * 100:.2f}%")

# Εμφάνιση της εξίσωσης του ορίου απόφασης
print(f"Εξίσωση Ορίου Απόφασης: {perceptron.weights[0]:.2f} * x1 + {perceptron.weights[1]:.2f} * x2 + {perceptron.bias:.2f} = 0")

# Οπτικοποίηση της εκπαίδευσης
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot των δεδομένων εκπαίδευσης
scatter_1 = ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='blue', edgecolor='k', label='Εκπαίδευση +1')
scatter_2 = ax[0].scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color='red', edgecolor='k', label='Εκπαίδευση -1')

# Plot των δεδομένων testing
scatter_3 = ax[0].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='cyan', label='Test +1', marker='x')
scatter_4 = ax[0].scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], color='orange', label='Test -1', marker='x')

# Όριο απόφασης
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
ax[0].set_xlim(x_min, x_max)
ax[0].set_ylim(y_min, y_max)
xx = np.linspace(x_min, x_max, 100)

if perceptron.weights[1] != 0:
    yy = -(perceptron.weights[0] * xx + perceptron.bias) / perceptron.weights[1]
    ax[0].plot(xx, yy, 'green', linestyle='--', label='Όριο Απόφασης')

# Χρωματισμός υποβάθρου
x_range = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 200)
x_range = np.linspace(x_min, x_max, 200)
y_range = np.linspace(y_min, y_max, 200)
xx_bg, yy_bg = np.meshgrid(x_range, y_range)
background = np.c_[xx_bg.ravel(), yy_bg.ravel()]
background = scaler.transform(background)
z = perceptron.predict(background).reshape(xx_bg.shape)
ax[0].contourf(xx_bg, yy_bg, z, alpha=0.2, levels=[-1, 0, 1], colors=['red', 'blue'], linestyles=['--'])

ax[0].legend()
ax[0].set_title('Όριο Απόφασης, Δεδομένα Εκπαίδευσης και Testing')

# Προσθήκη mplcursors για εμφάνιση τιμών με hover
cursor = mplcursors.cursor([scatter_1, scatter_2, scatter_3, scatter_4], hover=True)
@cursor.connect("add")
def on_add(sel):
    sel.annotation.set_text(f"x1: {sel.target[0]:.2f}\nx2: {sel.target[1]:.2f}")

# Plot του αριθμού λαθών ανά εποχή
ax[1].plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o', label='Λάθη ανά Εποχή')
ax[1].set_title('Λάθη ανά Εποχή')
ax[1].set_xlabel('Εποχή')
ax[1].set_ylabel('Αριθμός Λαθών')
ax[1].legend()

plt.tight_layout()
plt.show()
