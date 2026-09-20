# Contributing to AdaptNXT Jetson YOLO RTSP Pipeline

Thank you for your interest in contributing to the AdaptNXT Edge Computer Vision codebase! We welcome issues, bug reports, and pull requests.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat everyone with respect and kindness.

---

## Development Workflow

### 1. Fork & Clone
Fork the repository on GitHub and clone your fork locally:
```bash
git clone https://github.com/<your-username>/jetson-yolo-rtsp-pipeline.git
cd jetson-yolo-rtsp-pipeline
```

### 2. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```

### 3. Running Tests
```bash
pytest tests/ -v
```

### 4. Code Standards
- **Typing**: Use standard Python type annotations on all public functions and classes.
- **Hardware Agnostic**: Ensure core tests pass in mock mode without requiring NVIDIA GPU or CUDA runtime.
- **Docstrings**: Include descriptive docstrings detailing arguments, returns, and raised exceptions.

---

## Submitting Pull Requests

1. Create a descriptive branch: `git checkout -b feature/deepstream-python-binding`.
2. Commit your changes with conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`).
3. Push to your fork and open a Pull Request against `main`.
4. Maintainers from [AdaptNXT](https://www.adaptnxt.com) review PRs promptly.

---

## Questions & Support

For commercial support or custom NVIDIA Jetson Orin deployment, contact our engineering pod at [queries@adaptnxt.com](mailto:queries@adaptnxt.com).
