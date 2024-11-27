"""
This script categorizes books based on their descriptions using OpenAI.
"""

import csv
import argparse
import os
import time
import logging

import pandas as pd
import openai
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup


# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
open_ai_api = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API key
client = OpenAI(api_key=open_ai_api)


def load_categories_from_file(file_path) -> list[str]:
    """
    Load categories from a text file, where each category is listed on a new line.

    Args:
        file_path (str): The path to the text file containing categories.

    Returns:
        list: A list of categories.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            categories = [line.strip() for line in file if line.strip()]
        return categories
    except FileNotFoundError:
        logging.error("Category file not found at: %s", file_path)
        return []


# Function to clean HTML tags from the description
def clean_html(raw_html) -> str:
    """
    Remove HTML tags from a string.

    Args:
        raw_html (str): The string containing HTML.

    Returns:
        str: Cleaned text without HTML tags.
    """
    if pd.isna(raw_html):
        return ""

    # Parse HTML and extract text
    soup = BeautifulSoup(raw_html, "html.parser")
    cleaned_text = soup.get_text(separator=" ", strip=True)

    # Decode HTML entities like &#10; (newline) and others
    cleaned_text = cleaned_text.replace("\n", " ").replace("\r", " ")

    # Remove excessive whitespace, including encoded HTML spaces
    cleaned_text = " ".join(cleaned_text.split())

    # Explicitly remove all occurrences of "\n"
    cleaned_text = cleaned_text.replace("\\n", "").strip()

    # Return an empty string if the cleaned text is effectively empty
    if not cleaned_text:
        return ""

    return cleaned_text


# Function to get the category list from OpenAI API
def get_categories_for_book(description, categories_list) -> list[str]:
    """
    Fetch relevant categories for a book description using OpenAI API.

    Args:
        description (str): The book's description.
        categories_list (list): List of available categories.

    Returns:
        list: Selected categories relevant to the book description.
    """
    prompt = (
        f"Based on the following book description, choose the most relevant "
        f"categories from the provided list. Select between 3 and 10 categories "
        f"that best match the book's content. Make sure to only pick categories "
        f"that are clearly applicable, and avoid including irrelevant ones.\n\n"
        f"Description: {description}\n\n"
        f"Categories List: {', '.join(categories_list)}\n\n"
        f"Return the chosen categories as a comma-separated list without any additional text."
    )
    delay = 2  # Initial delay in seconds
    max_delay = 60  # Maximum delay in seconds
    max_retries = 5  # Maximum number of retry attempts

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
            )

            # Extract the text response
            categories_response = response.choices[0].message.content.strip()
            # Convert the comma-separated response to a list
            categories_list = [
                category.strip()
                for category in categories_response.split(",")
                if category.strip() in categories_list
            ]

            return categories_list
        except openai.RateLimitError:
            logging.warning(
                "Rate limit exceeded. Attempt %d/%d. Retrying in %d seconds.",
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
            delay = min(max_delay, delay * 2)  # Exponential backoff

        except openai.OpenAIError as e:
            logging.error("OpenAI API error: %s", e)
            return "No description available"

    # If all attempts fail
    logging.error("All retry attempts failed. Returning 'No description available'.")
    return "No description available"


# Validate and preprocess descriptions
def preprocess_description(ai_description, description, title):
    """
    Preprocess book descriptions by cleaning HTML and combining fields.

    Args:
        ai_description (str): AI-generated description.
        description (str): Original description.
        title (str): Title of the book.

    Returns:
        str: Cleaned and combined description.
    """
    # Prefer ai_description if available
    combined_description = ai_description if pd.notna(ai_description) else description

    # Clean HTML from the description
    combined_description = clean_html(combined_description)

    # Check if the cleaned description is empty
    if not combined_description.strip() or len(combined_description) < 5:
        logging.warning("No valid description found for title '%s'", title)
        return ""

    # Truncate the description if it's too long (let's say we limit it to 1000 characters)
    max_length = 1000
    if len(combined_description) > max_length:
        combined_description = combined_description[:max_length]
        logging.info("Truncated description for '%s' to %d characters.", title, max_length)

    combined_description = f"{title}: {combined_description}"

    return combined_description.strip()


def process_books(input_csv, output_csv, categories_file) -> None:
    """
    Process books by categorizing based on descriptions and saving results.

    Args:
        input_csv (str): Path to the input CSV file with book data.
        output_csv (str): Path to the output CSV file for results.
        categories_file (str): Path to the text file with categories.

    Returns:
        None
    """
    df = pd.read_csv(input_csv)
    categories = load_categories_from_file(categories_file)

    last_processed_id = get_last_processed_id(output_csv)

    if last_processed_id:
        start_index = df[df["id"] == last_processed_id].index[0] + 1
        df = df.iloc[start_index:]
        logging.info("Resuming processing from row index %d", start_index)

    with open(output_csv, "a", newline="", encoding="utf-8") as output_file:
        writer = setup_csv_writer(output_file)

        for _, row in df.iterrows():
            process_single_book(row, categories, writer, output_file)
    logging.info("Processing completed. Results are saved to %s", output_csv)


def get_last_processed_id(output_csv) -> str | None:
    """
    Get the last processed ID from the output CSV file.

    Args:
        output_csv (str): Path to the output CSV file.

    Returns:
        str: Last processed ID or None if no ID is found.
    """
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as output_file:
            reader = csv.DictReader(output_file)
            rows = list(reader)
            if rows:
                last_processed_id = rows[-1]["id"]
                logging.info("Last processed ID found: %s", last_processed_id)
                return last_processed_id
    return None


def setup_csv_writer(output_file) -> csv.DictWriter:
    """
    Set up a CSV writer for the output file.

    Args:
        output_file (file): Output file object.

    Returns:
        csv.DictWriter: CSV writer object.
    """
    fieldnames = ["id", "categories_list"]
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    if output_file.tell() == 0:
        writer.writeheader()
    return writer


def process_single_book(row, categories, writer, output_file) -> None:
    """
    Process a single book by categorizing and saving the results.

    Args:
        row (pd.Series): A row from the input DataFrame.
        categories (list): List of available categories.
        writer (csv.DictWriter): CSV writer object.
        output_file (file): Output file object.
    """
    book_id = row["id"]
    title = row.get("title", "")
    # ai_description = row.get("ai_description", None)
    description = row.get("description", "")
    tags = row.get("tags", "")

    logging.info("Processing book ID: %s", book_id)
    # combined_description = preprocess_description(tags, description, title)
    combined_description = f"{title}: {description} {tags}"
    if not combined_description:
        logging.warning("Skipping book ID %s due to lack of valid description.", book_id)
        return

    relevant_categories = get_categories_for_book(combined_description, categories)
    writer.writerow({"id": book_id, "categories_list": str(relevant_categories)})
    output_file.flush()


def main():
    """
    Main function to parse arguments and execute the book processing workflow.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Categorize books based on descriptions using OpenAI."
    )
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file with book data.")
    parser.add_argument(
        "output_csv", type=str, help="Path to the output CSV file for categorized data."
    )
    parser.add_argument("categories_file", type=str, help="Path to the categories text file.")
    args = parser.parse_args()

    process_books(args.input_csv, args.output_csv, args.categories_file)


if __name__ == "__main__":
    main()
