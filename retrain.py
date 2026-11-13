"""Rebuild the cleaned data and retrain the model in one command.

Run from the project root:
    python retrain.py
"""
from src import clean_data, train


def main() -> None:
    print("=== Step 1 of 2: cleaning ===")
    clean_data.main()
    print("\n=== Step 2 of 2: training ===")
    train.main()
    print("\nDone. Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
