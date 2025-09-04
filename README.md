# EconoVision Backend

This is the backend for the EconoVision project, built with Flask.

## Getting Started

1.  **Install Dependencies:** Make sure you have Python and `pip` installed. Then, from the `backend` directory, run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Server:** Start the Flask development server with:
    ```bash
    python app.py
    ```
    The server will run on `http://127.0.0.1:5000`.

## API Endpoints

-   `POST /api/upload`: Upload an Excel file.
-   `POST /api/analyze`: Run an economic analysis model (ARIMA, VAR, etc.).
-   `POST /api/report`: Generate a PDF report of the analysis results.
-   `GET /api/data`: Serves dummy data for the frontend.