# 🚀 AIAdapters: The Infinite Inference Engine for Open-NVR

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E.svg)](https://huggingface.co/models)

Welcome to the **AIAdapters** registry! This is a plug-and-play AI inference server built specifically for the **Open-NVR** ecosystem. 

**Our Goal:** To allow developers to drop *any* AI model into a surveillance network instantly—without modifying a single line of the core security code.

---

## 🌟 Why Build For AIAdapters?
Traditional NVRs (Network Video Recorders) lock you into their proprietary AI analytics. If you want to use a state-of-the-art model that dropped on Hugging Face yesterday, you can't.

**With AIAdapters, you can:**
- **Bring Your Own Model (BYOM):** Drop a new folder into `adapter/tasks/`, restart the server, and your custom analytic is live in the NVR UI.
- **Access 100,000+ Models instantly:** The built-in `HuggingFaceHandler` allows you to route streams directly to Hugging Face Cloud Inference endpoints securely. 
- **Mix & Match:** Run lightweight YOLOv11 person-counting locally on your edge device, while routing complex Face Recognition to a massive cloud GPU.

---

## 🤗 The HuggingFace Superpower
Want to use the latest BLIP-2 VQA model to ask your cameras questions? Or a specialized YOLO model for detecting PPE (hard hats)? 

You **do not need to code** an API integration. 
1. Get a token from HuggingFace.
2. Enter the Model ID in the Open-NVR UI.
3. The `HuggingFaceHandler` automatically ferries the isolated frames directly to the model, protecting your internal IP addresses while getting ultra-fast cloud inference.

---

## 🛠️ Quick Start (Developer Setup)

Get the inference engine running locally in 2 minutes:

```bash
# 1. Create virtual environment
uv venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate          # Windows

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Download base model weights (YOLO, etc.)
python download_models.py

# 4. Start the inference orchestration server
uvicorn adapter.main:app --reload --port 9100
```
*The server will boot, automatically discover all loaded plugins in the `tasks/` directory, and publish them to KAI-C.*

---

## 🧩 Contributing: Add Your Own AI Task!

We are building the largest open-source library of surveillance AI adapters. **We want your models!**

Adding a new capability is literally 3 steps:

1. **Create a Folder:** `mkdir adapter/tasks/wildlife_detection`
2. **Create the Contract:** Add a `schema.json` defining what the model returns (e.g., bounding boxes, confidence).
3. **Write the Handler:** Extend `BaseTask` and load your `.onnx` or `.pt` weights.

```python
# Example: adapter/tasks/wildlife_detection/task.py
from adapter.interfaces.task import BaseTask

class Task(BaseTask):
    def setup(self):
        self.handler = load_my_cool_model("bear_detector.onnx")
        
    def run(self, input_data):
        return self.handler.predict(input_data)
```

**That's it.** The `loader.py` engine dynamically inherits your class and securely binds it to a live HTTP endpoint without risking the main server going down.

See [docs/PLUGIN_DEVELOPMENT.md](docs/PLUGIN_DEVELOPMENT.md) for the full tutorial.

---

## 📂 Project Architecture

```
AIAdapters/
├── adapter/                    # The Engine Loop
│   ├── main.py                 # FastAPI Router
│   ├── loader.py               # Auto-discovers plugins at boot
│   ├── interfaces/             # The BaseTask contract
│   │
│   ├── tasks/                  # 🔥 PLUGIN FOLDER 🔥
│   │   ├── person_counting/
│   │   ├── scene_description/
│   │   └── <YOUR_NEW_TASK>/    # Drop your folder here!
│   │
│   └── models/                 # Model interaction logic
│       ├── yolov11_handler.py  # YOLOv11 + ByteTrack PyTorch
│       └── huggingface_handler.py # Cloud orchestration
│
├── docs/                       # Developer Documentation
├── model_weights/              # Downloaded .onnx files
└── Dockerfile                  # GPU and CPU build files
```

---

## 🤝 Join the Community
By contributing an Adapter, you help secure and democratize AI for edge environments globally. 
Submit a Pull Request today! If your model passes validation, it will be merged into the official Open-NVR release.

---

## ⚖️ License
The AI Adapters framework, plugins, and handlers are licensed strictly under the **GNU Affero General Public License v3.0 (AGPL v3)**. 
All forks, internal cloud pipelines, and customized models running atop this core engine must share their source code to the OpenNVR ecosystem under the same AGPL terms. See the `LICENSE` file for the binding legal outline.

> For commercial usage exemptions, proprietary adapter development, or direct enterprise support, please contact: **[contact@cryptovoip.in](mailto:contact@cryptovoip.in)**
