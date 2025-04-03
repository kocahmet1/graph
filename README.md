# Veritas AI Tools

A Flask web application that uses Google's Gemini AI to generate graphs and process questions from images.

## Features

- Generate graphs from natural language descriptions
- Process uploaded question images to create new questions of the same type
- Toggle between fast and slow model preference
- Customizable Google Gemini API key

## Deployment to Render

### Prerequisites

1. A [Render](https://render.com/) account
2. Google Gemini API key

### Deployment Steps

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the service:
   - **Name**: veritas-ai-tools (or your preferred name)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

4. Add Environment Variables:
   - `GOOGLE_API_KEY`: Your Google Gemini API key

5. Deploy the service

### Post-Deployment

After deployment, you'll need to create a `config.json` file on Render with your API key. You can do this through the Render Shell:

1. Go to your Web Service on Render
2. Open the Shell tab
3. Run the following command:

```bash
echo '{"GOOGLE_API_KEY": "your-gemini-api-key-here", "MODEL_PREFERENCE": "slow"}' > config.json
```

## Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `config.json` file in the root directory with your Google Gemini API key:
```json
{
  "GOOGLE_API_KEY": "your-gemini-api-key-here",
  "MODEL_PREFERENCE": "slow"
}
```
4. Run the application: `python app.py`
5. Access the application at http://localhost:5000
