# Προσπάθεια εισαγωγής βιβλιοθηκών οπτικοποίησης
try:
    import matplotlib.pyplot as plt
    import mplcursors

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print(
        "Προειδοποίηση: Οι βιβλιοθήκες οπτικοποίησης δεν βρέθηκαν. Χρήση εξόδου τερματικού.\n"
    )


def linspace(start, stop, num):
    """Υλοποίηση της numpy linspace"""
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.errors = []

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0

        for epoch in range(self.epochs):
            error_count = 0
            for i in range(len(X)):
                linear_output = (
                    sum(x_j * w_j for x_j, w_j in zip(X[i], self.weights)) + self.bias
                )
                y_predicted = 1 if linear_output >= 0 else -1

                if y_predicted != y[i]:
                    update = self.learning_rate * y[i]
                    self.weights = [w + update * x for w, x in zip(self.weights, X[i])]
                    self.bias += update
                    error_count += 1

            self.errors.append(error_count)

        # Εκτύπωση πληροφοριών για την τρέχουσα εποχή
        print_error_plot(perceptron.errors)

    def predict(self, X):
        predictions = []
        for x in X:
            linear_output = (
                sum(x_j * w_j for x_j, w_j in zip(x, self.weights)) + self.bias
            )
            predictions.append(1 if linear_output >= 0 else -1)
        return predictions


def load_data(filename):
    X, y = [], []
    with open(filename, "r") as f:
        next(f)  # Παράλειψη κεφαλίδας
        for line in f:
            x1, x2, label = map(float, line.strip().split(","))
            X.append([x1, x2])
            y.append(1 if label == 1.0 else -1)
    return X, y


def get_min_max(data, column):
    values = [row[column] for row in data]
    return min(values), max(values)


def print_decision_boundary(X, y, perceptron, width=50, height=20):
    """Οπτικοποίηση ορίου απόφασης στο τερματικό"""
    if not VISUALIZATION_AVAILABLE:
        x_min, x_max = get_min_max(X, 0)
        y_min, y_max = get_min_max(X, 1)

        # Δημιουργία αναπαράστασης ASCII
        print("\nΌριο Απόφασης:")
        print("-" * (width + 2))

        for i in range(height):
            y_val = y_max - (i * (y_max - y_min) / height)
            line = "|"
            for j in range(width):
                x_val = x_min + (j * (x_max - x_min) / width)
                prediction = (
                    1
                    if (
                        perceptron.weights[0] * x_val
                        + perceptron.weights[1] * y_val
                        + perceptron.bias
                    )
                    >= 0
                    else -1
                )
                line += "+" if prediction == 1 else "-"
            line += "|"
            print(line)
        print("-" * (width + 2))
        print(f"Υπόμνημα: '+' = θετική περιοχή, '-' = αρνητική περιοχή")


def print_error_plot(errors):
    """Οπτικοποίηση εξέλιξης σφαλμάτων στο τερματικό"""
    print("Εξέλιξη Σφαλμάτων:")
    max_error = max(errors)
    width = 40
    for i, error in enumerate(errors, 1):
        bar_length = int((error / max_error) * width)
        print(f"Εποχή {i:2d}: {'█' * bar_length} ({error})")


# Φόρτωση δεδομένων
X_train, y_train = load_data("training_data.csv")
X_test, y_test = load_data("test_data.csv")

# Εκπαίδευση perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=20)
perceptron.fit(X_train, y_train)

# Έλεγχος ακρίβειας
y_pred = perceptron.predict(X_test)
accuracy = sum(1 for y1, y2 in zip(y_pred, y_test) if y1 == y2) / len(y_test)

# Εκτύπωση βασικών πληροφοριών (ανεξάρτητα από τη διαθεσιμότητα οπτικοποίησης)
print("\nΑποτελέσματα Εκπαίδευσης:")
print(f"Τελικά βάρη: {perceptron.weights}")
print(f"Τελική πόλωση: {perceptron.bias}")
print(f"Ακρίβεια Τεστ: {accuracy * 100:.2f}%")
print(
    f"Εξίσωση ορίου απόφασης: {perceptron.weights[0]:.2f}x₁ + {perceptron.weights[1]:.2f}x₂ + ({perceptron.bias:.2f}) = 0"
)

# Modified visualization section
if VISUALIZATION_AVAILABLE:
    # Οπτικοποίηση
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Λήψη εύρους δεδομένων
    x_min, x_max = get_min_max(X_train, 0)
    y_min, y_max = get_min_max(X_train, 1)
    x_min, x_max = x_min - 1, x_max + 1
    y_min, y_max = y_min - 1, y_max + 1

    # Σχεδίαση δεδομένων εκπαίδευσης
    train_pos = [(x[0], x[1]) for x, y in zip(X_train, y_train) if y == 1]
    train_neg = [(x[0], x[1]) for x, y in zip(X_train, y_train) if y == -1]

    scatter_1 = ax[0].scatter(
        [x[0] for x in train_pos],
        [x[1] for x in train_pos],
        color="blue",
        edgecolor="k",
        label="Εκπαίδευση +1",
    )
    scatter_2 = ax[0].scatter(
        [x[0] for x in train_neg],
        [x[1] for x in train_neg],
        color="red",
        edgecolor="k",
        label="Εκπαίδευση -1",
    )

    # Σχεδίαση δεδομένων ελέγχου
    test_pos = [(x[0], x[1]) for x, y in zip(X_test, y_test) if y == 1]
    test_neg = [(x[0], x[1]) for x, y in zip(X_test, y_test) if y == -1]

    scatter_3 = ax[0].scatter(
        [x[0] for x in test_pos],
        [x[1] for x in test_pos],
        color="cyan",
        marker="x",
        label="Τεστ +1",
    )
    scatter_4 = ax[0].scatter(
        [x[0] for x in test_neg],
        [x[1] for x in test_neg],
        color="orange",
        marker="x",
        label="Τεστ -1",
    )

    # Σχεδίαση ορίου απόφασης και γέμισμα φόντου
    xx = linspace(x_min, x_max, 100)
    if perceptron.weights[1] != 0:
        yy = [
            -(perceptron.weights[0] * x + perceptron.bias) / perceptron.weights[1]
            for x in xx
        ]

        # Προσθήκη χρωμάτων φόντου
        ax[0].fill_between(
            xx, yy, [y_max] * len(xx), alpha=0.2, color="blue", label="Περιοχή +1"
        )
        ax[0].fill_between(
            xx, yy, [y_min] * len(xx), alpha=0.2, color="red", label="Περιοχή -1"
        )

        # Σχεδίαση γραμμής ορίου απόφασης
        ax[0].plot(xx, yy, "green", linestyle="--", label="Όριο Απόφασης")

        # Προσθήκη κειμένου εξίσωσης υπό γωνία
        equation = f"{perceptron.weights[0]:.2f}x₁ + {perceptron.weights[1]:.2f}x₂ + ({perceptron.bias:.2f}) = 0"
        text_x = x_min + (x_max - x_min) * 0.675
        text_y = y_min + (y_max - y_min) * 0.027
        ax[0].text(
            text_x,
            text_y,
            equation,
            rotation=-35,
            color="green",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)
    # Τοποθέτηση υπομνήματος στην πάνω αριστερή γωνία
    ax[0].legend(loc="upper left")
    ax[0].set_title("Όριο Απόφασης, Εκπαίδευση και Τεστ")

    # Προσθήκη mplcursors για εμφάνιση στο hover
    cursor = mplcursors.cursor([scatter_1, scatter_2, scatter_3, scatter_4], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(f"x1: {sel.target[0]:.2f}\nx2: {sel.target[1]:.2f}")

    # Σχεδίαση λαθών ανά εποχή
    ax[1].plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker="o")
    ax[1].set_title("Λάθη ανά Εποχή")
    ax[1].set_xlabel("Εποχή")
    ax[1].set_ylabel("Αριθμός Λαθών")

    plt.tight_layout()
    plt.show()
else:
    # Εκτύπωση οπτικοποιήσεων τερματικού
    print_decision_boundary(X_train, y_train, perceptron)
