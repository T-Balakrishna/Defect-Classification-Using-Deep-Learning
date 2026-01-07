from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# -----------------------------
# Model Setup (unchanged)
# -----------------------------
num_classes = 6
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("smt/defect_model.pth", map_location="cpu"))
model.eval()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Classes
# -----------------------------
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# -----------------------------
# Helper: PIL → base64
# -----------------------------
def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -----------------------------
# Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]  # ← FIXED
        if file:
            # Your original logic
            filename = file.filename.split('_')[0]
            print(filename)
            # Open image from stream
            img = Image.open(file.stream).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)

            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, 1)
                result = classes[pred.item()]

            # Encode image for display
            image_data = pil_to_base64(img)

            return render_template("index.html",
                                   prediction=filename,
                                   image_data=image_data)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)