# Adaptive Test System

This project is an adaptive testing platform for Math proficiency, powered by Item Response Theory (IRT) and Computerized Adaptive Testing (CAT). It dynamically selects questions based on the user's ability, providing a personalized and efficient assessment experience.

## Features

- **Adaptive Testing:** Questions adapt in real time to the user's skill level.
- **Item Response Theory (IRT):** Uses IRT models to estimate ability (θ) and select the most informative questions.
- **Comprehensive Question Bank:** Over 5,000 items covering basic mathematics 
- **User Dashboard:** Visualizes progress, ability estimates, and test statistics.
- **Session Management:** User progress is saved between sessions.
- **Modern UI:** Responsive and user-friendly interface.


# Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
    ```sh
    git clone <repo-url>
    cd Adaptive\ Learning
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **(Optional) Set up a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```

### Running the App

1. **Start the Flask server:**
    ```sh
    cd frontend
    python app.py
    ```

2. **Open your browser and go to:**
    ```
    http://127.0.0.1:5000/
    ```

## Usage

- Enter your name and start the test.
- Answer the questions; your ability estimate (θ) updates after each response.
- View your progress and statistics on the dashboard.
- Continue the test or log out as needed.

# Project Structure

## File Descriptions

- [`frontend/app.py`](frontend/app.py): Main Flask application.
- [`frontend/CAT_egine.py`](frontend/CAT_egine.py): Adaptive testing engine using IRT.
- [`frontend/templates/`](frontend/templates/): HTML templates for UI.
- [`frontend/item_bank.csv`](frontend/item_bank.csv): Item parameters for adaptive selection.
- [`frontend/questions.csv`](frontend/questions.csv): Question text and answer options.
- [`Developing codes/`](Developing codes/): Data processing, simulation, and experimental scripts.

## Customization

- **Question Bank:** Update `frontend/questions.csv` to add or modify questions.
- **Item Bank:** Update `frontend/item_bank.csv` for new items or recalibrated parameters.
- **UI:** Edit templates in `frontend/templates/` for branding or layout changes.

## License

This project is for educational and research purposes.

---

*For questions or contributions, please open an issue or