# Agent-tool-app
Streamlit Cloud-ready Web App for Prompt Engineering

### 1. Repository Description
> Related Project: OpenSource Project  
> Assignees: Jaehyeong Baek, Seonguk Choi  
> Description: A web app that helps with prompt engineering on the UI.

### 2. Features
- **Multimodal Support**: Supports text, image, audio, and video inputs for LLM inference.
- **Time Series Support**: Custom logic for time series rule base inference.
- **Risk Analysis**: Merging and analyzing risk data.
- **Provider Support**: Google Gemini and OpenAI.

### 3. Execution
To run locally:
```bash
# Install requirements
pip install -r requirements.txt

# Run the app
./run/start_app.sh
# OR
streamlit run app.py --server.port 8989
```

### 4. Streamlit Cloud Deployment
This repository is ready to be deployed on Streamlit Cloud.
1. Push this repository to GitHub.
2. Go to [Streamlit Share](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app** -> **Deploy a public app from GitHub**.
4. Select this repository, branch, and set the **Main file path** to `app.py`.
5. Under **Advanced settings**, set your environment variables (e.g. `GEMINI_API_KEY`, `OPENAI_API_KEY`, etc.).
6. Click **Deploy!**

### 5. Folder Structure
```
.
├── configs/
│   ├── __init__.py
│   └── config.py          # App settings and environment variables
├── prompts/               # System and User Prompts
├── run/
│   └── start_app.sh       # Execution script
├── src/
│   ├── __init__.py
│   ├── llm_manager.py     # LLM Inference logic (OpenAI/Gemini)
│   ├── ui_components.py   # Sidebar and Upload section UI
│   ├── utils.py           # Utility functions
│   └── time_series_rule_base_logic.py
├── requirements.txt
├── app.py                 # Main Streamlit app (Entry Point)
└── README.md
```