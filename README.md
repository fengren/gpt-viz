# Transformer ChatGPT Visual

A minimal version of ChatGPT based on Transformer models, with complete processing flow visualization.

## Features

### Basic Features
- 🔤 Tokenization: Breaking continuous text into meaningful tokens
- 🔢 Encoding: Converting tokens to numerical IDs
- 📊 Vectorization: Transforming token IDs to continuous vector representations
- 📏 Normalization: Scaling vectors to a fixed norm
- 🔗 Correlation Calculation: Measuring similarity between texts
- 🏷️ Text Classification: Classifying texts based on vector representations

### Advanced Features
- 🧠 MCP (Model Context Processing): Optimizing model inputs for better understanding
- 🛠️ Skill (Tool Calling): Allowing models to use external tools and APIs
- 🔍 RAG (Retrieval-Augmented Generation): Combining information retrieval with generative AI

### Visualization Features
- 📊 Real-time update loading progress bar
- 👁️ Attention relationship heatmap
- 📈 Softmax activation function visualization
- 🔢 Dot product calculation visualization

## Quick Start

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the Application

```bash
python -m streamlit run visual_transformer.py
```

Then access `http://localhost:8501` in your browser

## Deploy to Vercel

### Preparation

1. Create a GitHub repository and push the code
2. Create a new project on Vercel and link it to your GitHub repository
3. Add the following secrets in GitHub repository's `Settings > Secrets and variables > Actions`:
   - `VERCEL_TOKEN`: Obtained from Vercel account settings
   - `VERCEL_ORG_ID`: Obtained from Vercel project settings
   - `VERCEL_PROJECT_ID`: Obtained from Vercel project settings

### Automatic Deployment

When you push code to the `main` branch, GitHub Action will automatically trigger the deployment process to deploy the application to Vercel.

### Manual Deployment

You can also deploy manually using Vercel CLI:

```bash
npm install -g vercel
vercel login
vercel deploy --prod
```

## Project Structure

```
transformer_demo/
├── visual_transformer.py   # Main application file
├── simple_implementation.py # Simple command-line implementation
├── requirements.txt        # Project dependencies
├── vercel.json             # Vercel configuration file
├── .github/workflows/
│   └── deploy.yml          # GitHub Action workflow
├── README.md               # English project documentation
└── README_zh.md            # Chinese project documentation
```

## Tech Stack

- **Framework**: Streamlit
- **Model**: BERT-base-multilingual-cased
- **Libraries**: Transformers, PyTorch, NumPy, Scikit-learn, Matplotlib
- **Deployment**: Vercel, GitHub Actions

## Usage Instructions

1. Enter query content in the sidebar (maximum 200 characters)
2. Select the processing steps you want to demonstrate
3. Check "Enable Advanced Features" to use MCP, Skill, and RAG features
4. Click the "Process" button
5. Expand each step to view detailed processing and visualizations

## Notes

- It may take some time to download the pre-trained model for the first run
- Some advanced features require significant computational resources
- When deploying on Vercel, you may need to adjust the model size or use a more powerful runtime

## License

MIT
