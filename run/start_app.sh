#!/bin/bash
# start_app.sh: Run the Streamlit Unified Labeling App from root

# Set PYTHONPATH to root to resolve absolute imports
export PYTHONPATH=$PYTHONPATH:.

# Run Streamlit
streamlit run app.py --server.port 8003
