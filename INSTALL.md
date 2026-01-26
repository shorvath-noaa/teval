# Installation Instructions

## 🚀 Quick Start (For Users)
If you just want to use the tool to analyze data:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/teval.git](https://github.com/YOUR_USERNAME/teval.git)
   cd teval
   ```

2. **Create a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install .
    ```

## 🛠️ Developer Setup (For Contributors)
If you want to edit the code, run tests, or contribute features:

1. **Clone and Setup Environment:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/teval.git](https://github.com/YOUR_USERNAME/teval.git)
    cd teval
    python -m venv .venv
    source .venv/bin/activate
    ```

2. **Install in Editable Mode:** This allows you to change code and see results immediately without reinstalling.
    ```bash
    pip install -e .[dev]
    ```

3. **Verify Installation:** Run the test suite to ensure everything is working:
    ```bash
    pytest
    ```