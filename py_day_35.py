class Model:
    def __init__(self, name):
        self.name = name
        self.trained = False

    def train(self):
        self.trained = True
        print(f"{self.name} training complete.")

# Classifier inherits from Model
class Classifier(Model):
    def predict(self, data):
        if not self.trained:
            print("Warning: Model not trained.")
        return [0] * len(data)

clf = Classifier("clf")
clf.train()
print(clf.predict(["I love AI"]))   # [0]