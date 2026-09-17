# 🚀 Production Deployment Guide

This guide details how to deploy the **Student Performance Prediction & Learning Intelligence System** to various cloud hosting platforms for free or in production environments.

---

## ⚡ Method 1: Deploy on Vercel (Fast & Global Serverless Edge)

Deploy both the interactive Web Dashboard and the FastAPI REST API Serverless Functions instantly on Vercel.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Add Vercel deployment configuration"
   git push origin main
   ```
2. Go to **[vercel.com](https://vercel.com/)** and sign in with GitHub.
3. Click **"Add New..."** $\rightarrow$ **"Project"**.
4. Import your GitHub repository: `Student-Performance-Prediction-System-`.
5. Keep Framework Preset as **"Other"** (Vercel automatically detects `vercel.json` and `api/index.py`).
6. Click **"Deploy"**.
7. Once deployed (~1-2 minutes), you get a fast global HTTPS URL:
   - 🌐 **Web Dashboard:** `https://your-project.vercel.app/`
   - 📑 **API Documentation:** `https://your-project.vercel.app/docs`
   - 📊 **Predictions Endpoint:** `https://your-project.vercel.app/api/v1/predict`

---

## 🌟 Method 2: Deploy on Render (Recommended for Full Containers)

Render provides free hosting with automatic HTTPS, continuous deployment from GitHub, and WebSocket/REST support.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Add production deployment configs"
   git push origin main
   ```
2. Go to **[render.com](https://render.com/)** and sign in with GitHub.
3. Click **"New +"** $\rightarrow$ **"Web Service"**.
4. Connect your GitHub repository: `Student-Performance-Prediction-System-`.
5. Render will automatically detect `render.yaml` or you can manually configure:
   - **Name**: `student-performance-prediction`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
6. Click **"Create Web Service"**.
7. Once deployed (typically ~2-3 minutes), your live URL will be:
   - 🌐 Dashboard: `https://your-app-name.onrender.com/dashboard/index.html`
   - 📑 API Docs: `https://your-app-name.onrender.com/docs`

---

## 🎈 Method 2: Deploy on Streamlit Community Cloud (Free)

Ideal for hosting the Streamlit Enterprise Dashboard (`streamlit_app.py`).

### Steps:
1. Push your repository to GitHub.
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Select:
   - **Repository**: `Your-Username/Student-Performance-Prediction-System-`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click **"Deploy!"**.

---

## 🤗 Method 4: Deploy on Hugging Face Spaces (Free)

Hugging Face Spaces offers free CPU hardware and instant public sharing.

### Steps:
1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)**.
2. Click **"Create new Space"**.
3. Choose:
   - **Space SDK**: `Docker` (or `Streamlit` for `streamlit_app.py`)
   - **License**: `MIT` or `Open Source`
4. Clone the space or connect your GitHub repo and push the files. Hugging Face will automatically build the `Dockerfile` and launch the app!

---

## 🐳 Method 5: Deploy with Docker (AWS / GCP / DigitalOcean / VPS)

You can run the entire system on any cloud server using Docker:

### Build and Run Locally:
```bash
# Build the Docker image
docker build -t student-performance-prediction .

# Run container on port 8000
docker run -d -p 8000:8000 --name student-ai student-performance-prediction
```
Open [http://localhost:8000/dashboard/index.html](http://localhost:8000/dashboard/index.html).

### Run Multi-Service with Docker Compose:
```bash
# Launches both FastAPI (port 8000) and Streamlit (port 8501)
docker compose up -d
```

### Deploy to GCP Cloud Run (1 Command):
```bash
gcloud run deploy student-performance \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### Deploy to AWS App Runner / ECS:
1. Push image to Amazon ECR:
   ```bash
   aws ecr create-repository --repository-name student-performance
   docker tag student-performance-prediction:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/student-performance:latest
   docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/student-performance:latest
   ```
2. Create an **App Runner Service** pointing to the ECR image on port `8000`.

---

## 🛠️ Verification Checklist After Deployment

| Checkpoint | Target URL | Expected Result |
| :--- | :--- | :--- |
| **Health Check** | `/health` | `{"status": "healthy", "model_loaded": true}` |
| **Web UI** | `/dashboard/index.html` | Interactive 2026 AI Dashboard renders with real-time prediction sliders |
| **API Docs** | `/docs` | Interactive Swagger UI allows live POST `/api/v1/predict` testing |
| **Metadata** | `/api/v1/features` | Returns selected features, model names, and target classes |
