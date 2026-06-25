# 🫁 Chest X-Ray Pneumonia Detector

A Streamlit web app that classifies chest X-ray images as **Normal** or **Pneumonia** using a fine-tuned **DenseNet121** model.

**Model:** DenseNet121 (Transfer Learning)  
**Test Accuracy:** 91%  
**Framework:** TensorFlow / Keras  
**Dataset:** Chest X-Ray Images (Kaggle)

---

## 📁 Folder Structure

```
chest-xray-app/
├── app.py
├── requirements.txt
├── README.md
└── best_densenet_pneumonia.keras   ← your trained model (add this!)
```

---

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
1. Create a new GitHub repo (e.g. `chest-xray-pneumonia-streamlit`)
2. Upload these files: `app.py`, `requirements.txt`, `README.md`
3. Upload your trained model file: `best_densenet_pneumonia.keras`

> ⚠️ If model file > 100 MB, use [Git LFS](https://git-lfs.com/) or host it on Google Drive / Hugging Face Hub and load it programmatically.

### Step 2 — Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy!**

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `best_densenet_pneumonia.keras` in the same folder before running.

---

## 📦 Model File — Large File Tip

If your `.keras` file is > 100 MB, upload it to **Hugging Face Hub** and load it like this in `app.py`:

```python
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="your-username/your-repo",
        filename="best_densenet_pneumonia.keras"
    )
    return tf.keras.models.load_model(path)
```

Then add `huggingface_hub` to `requirements.txt`.

---

Built by **Nupur Zile** | AI Developer & Data Scientist
