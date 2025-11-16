 Smart Waste Classifier

A small service and web UI for classifying waste (plastic, paper, organic, etc.) using a local ML model.

## Features
- Upload images and get predicted waste category
- Local ONNX model support
- Simple web UI with Tailwind CSS
- Configurable via `.env`

## Prerequisites
- Node.js 16+ (or compatible)
- npm, yarn or pnpm
- Git (optional)
- Model file (ONNX) placed at the path configured in `.env`

##  Install dependencies
````powershell
npm install
# or
# yarn
# pnpm install

## Structure
Project layout (important files)
app/ — frontend assets (Tailwind, CSS)
models/ — place your ONNX model here (matches MODEL_PATH)
uploads/ — temporary uploaded images
.env — runtime configuration (do not commit)

##Run the Application
npm run dev
# or
npm start
