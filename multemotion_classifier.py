import torch
from torch import nn

# Define the emotion classifier model
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionClassifier, self).__init__()
        self.embed = nn.Embedding(512, 16)  # ASCII range
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.embed(x).mean(dim=1)
        return self.fc(x)

# Initialize the model
model = EmotionClassifier(num_classes=8)

# Here, you can load a pre-trained model if you have one
# model.load_state_dict(torch.load("emotion_model.pth"))

# Available emotion labels
emotion_labels = ["happy", "sad", "angry", "surprised", "confused", "excited", "nervous", "bored"]

# Prompt asking the user to enter text
print("Enter text for emotion analysis:")

while True:
    input_text = input(">> ")  # Prompt for user input
    if input_text.lower() == 'exit':  # Exit if the user types "exit"
        print("Program terminated.")
        break

    # Convert input text to ASCII values and pad to a specified length
    input_tensor = torch.tensor([ord(c) if ord(c) < 512 else 0 for c in input_text], dtype=torch.long)
    padded_input_tensor = torch.zeros(30, dtype=torch.long)  # Maximum length is 30
    padded_input_tensor[:len(input_tensor)] = input_tensor[:30]
    padded_input_tensor = padded_input_tensor.unsqueeze(0)

    # Predict the emotion
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        prediction = model(padded_input_tensor)
        predicted_label = torch.argmax(prediction, dim=1).item()

    # Print the result
    print("Emotion:", emotion_labels[predicted_label])
