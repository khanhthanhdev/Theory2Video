---
title: AI Animation & Voice Studio
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
suggested_hardware: cpu-upgrade
suggested_storage: large
pinned: true
license: apache-2.0
short_description: "Create mathematical animations with AI-powered using Manim"
tags:
  - text-to-speech
  - animation
  - mathematics
  - manim
  - ai-voice
  - educational
  - visualization
models:
  - kokoro-onnx/kokoro-v0_19
datasets: []
startup_duration_timeout: 30m
fullWidth: true
header: default
disable_embedding: false
preload_from_hub: []
---

# AI Animation & Voice Studio 🎬

A powerful application that combines AI-powered text-to-speech with mathematical animation generation using Manim and Kokoro TTS. Create stunning educational content with synchronized voice narration and mathematical visualizations.

## 🚀 Features

- **Text-to-Speech**: High-quality voice synthesis using Kokoro ONNX models
- **Mathematical Animations**: Create stunning mathematical visualizations with Manim
- **LaTeX Support**: Full LaTeX rendering capabilities with TinyTeX
- **Interactive Interface**: User-friendly Gradio web interface
- **Audio Processing**: Advanced audio manipulation with FFmpeg and SoX

## 🛠️ Technology Stack

- **Frontend**: Gradio for interactive web interface
- **Backend**: Python with FastAPI/Flask
- **Animation**: Manim (Mathematical Animation Engine)
- **TTS**: Kokoro ONNX for text-to-speech synthesis
- **LaTeX**: TinyTeX for mathematical typesetting
- **Audio**: FFmpeg, SoX, PortAudio for audio processing
- **Deployment**: Docker container optimized for Hugging Face Spaces

## 📦 Models

This application uses the following pre-trained models:

- **Kokoro TTS**: `kokoro-v0_19.onnx` - High-quality neural text-to-speech model
- **Voice Models**: `voices.bin` - Voice embedding models for different speaker characteristics

Models are automatically downloaded during the Docker build process from the official releases.

## 🏃‍♂️ Quick Start

### Using Hugging Face Spaces

1. Visit the [Space](https://huggingface.co/spaces/your-username/ai-animation-voice-studio)
2. Wait for the container to load (initial startup may take 3-5 minutes due to model loading)
3. Upload your script or enter text directly
4. Choose animation settings and voice parameters
5. Generate your animated video with AI narration!

### Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/your-username/ai-animation-voice-studio
cd ai-animation-voice-studio

# Build the Docker image
docker build -t ai-animation-studio .

# Run the container
docker run -p 7860:7860 ai-animation-studio
```

Access the application at `http://localhost:7860`

### Environment Setup

Create a `.env` file with your configuration:

```env
# Application settings
DEBUG=false
MAX_WORKERS=4

# Model settings
MODEL_PATH=/app/models
CACHE_DIR=/tmp/cache

# Optional: API keys if needed
# OPENAI_API_KEY=your_key_here
```

## 🎯 Usage Examples

### Basic Text-to-Speech

```python
# Example usage in your code
from src.tts import generate_speech

audio = generate_speech(
    text="Hello, this is a test of the text-to-speech system",
    voice="default",
    speed=1.0
)
```

### Mathematical Animation

```python
# Example Manim scene
from manim import *

class Example(Scene):
    def construct(self):
        # Your animation code here
        pass
```

## 📁 Project Structure

```
├── src/                    # Source code
│   ├── tts/               # Text-to-speech modules
│   ├── manim_scenes/      # Manim animation scenes
│   └── utils/             # Utility functions
├── models/                # Pre-trained models (auto-downloaded)
├── output/                # Generated content output
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── gradio_app.py         # Main application entry point
└── README.md             # This file
```

## ⚙️ Configuration

### Docker Environment Variables

- `GRADIO_SERVER_NAME`: Server host (default: 0.0.0.0)
- `GRADIO_SERVER_PORT`: Server port (default: 7860)
- `PYTHONPATH`: Python path configuration
- `HF_HOME`: Hugging Face cache directory

### Application Settings

Modify settings in your `.env` file or through environment variables:

- Model parameters
- Audio quality settings
- Animation render settings
- Cache configurations

## 🔧 Development

### Prerequisites

- Docker and Docker Compose
- Python 3.12+
- Git

### Setting Up Development Environment

```bash
# Install dependencies locally for development
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Format code
black .
isort .

# Lint code
flake8 .
```

### Building and Testing

```bash
# Build the Docker image
docker build -t your-app-name:dev .

# Test the container locally
docker run --rm -p 7860:7860 your-app-name:dev

# Check container health
docker run --rm your-app-name:dev python -c "import src; print('Import successful')"
```

## 📊 Performance & Hardware

### Recommended Specs for Hugging Face Spaces

- **Hardware**: `cpu-upgrade` (recommended for faster rendering)
- **Storage**: `small` (sufficient for models and temporary files)
- **Startup Time**: ~3-5 minutes (due to model loading and TinyTeX setup)
- **Memory Usage**: ~2-3GB during operation

### System Requirements

- **Memory**: Minimum 2GB RAM, Recommended 4GB+
- **CPU**: Multi-core processor recommended for faster animation rendering
- **Storage**: ~1.5GB for models and dependencies
- **Network**: Stable connection for initial model downloads

### Optimization Tips

- Models are cached after first download
- Gradio interface uses efficient streaming for large outputs
- Docker multi-stage builds minimize final image size
- TinyTeX installation is optimized for essential packages only

## 🐛 Troubleshooting

### Common Issues

**Build Failures**:
```bash
# Clear Docker cache if build fails
docker system prune -a
docker build --no-cache -t your-app-name .
```

**Model Download Issues**:
- Check internet connection
- Verify model URLs are accessible
- Models will be re-downloaded if corrupted

**Memory Issues**:
- Reduce batch sizes in configuration
- Monitor memory usage with `docker stats`

**Audio Issues**:
- Ensure audio drivers are properly installed
- Check PortAudio configuration

### Getting Help

1. Check the [Discussions](https://huggingface.co/spaces/your-username/ai-animation-voice-studio/discussions) tab
2. Review container logs in the Space settings
3. Enable debug mode in configuration
4. Report issues in the Community tab

### Common Configuration Issues

**Space Configuration**:
- Ensure `app_port: 7860` is set in README.md front matter
- Check that `sdk: docker` is properly configured
- Verify hardware suggestions match your needs

**Model Loading**:
- Models download automatically on first run
- Check Space logs for download progress
- Restart Space if models fail to load

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use Black for code formatting
- Add docstrings for functions and classes
- Include type hints where appropriate

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Manim Community](https://www.manim.community/) for the animation engine
- [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) for text-to-speech models
- [Gradio](https://gradio.app/) for the web interface framework
- [Hugging Face](https://huggingface.co/) for hosting and infrastructure

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)
- **Hugging Face**: [@your-username](https://huggingface.co/your-username)

---

*Built with ❤️ for the open-source community*