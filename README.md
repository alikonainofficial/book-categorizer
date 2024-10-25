# Book Categorization Script

This script categorizes books based on their descriptions by utilizing the OpenAI API. It takes a CSV file with book information, analyzes the descriptions, and outputs relevant categories based on a predefined list of categories.

## Features

- Cleans and preprocesses book descriptions, removing any HTML content.
- Uses OpenAI to suggest categories for each book description.
- Supports resuming from the last processed book in case of interruptions.
- Saves results in a CSV file with each book's categories.

## Requirements

- Python 3.7 or above
- Dependencies listed in `requirements.txt`:
  - `openai`
  - `python-dotenv`
  - `beautifulsoup4`
  - `pandas`

## Setup

1. **Clone the repository** (if applicable) and navigate to the project folder.
   
2. **Install dependencies** by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   - Create a .env file in the project root.
   - Add your OpenAI API key:

               
         OPENAI_API_KEY=your_api_key_here
               

4. **Prepare category list**:
   - Create a text file containing book categories, each category on a new line.
   - Provide the path to this file when running the script.

## Usage

Run the script with the following command:

```bash
python script_name.py <input_csv> <output_csv> <categories_file>
```
- `<input_csv>`: Path to the CSV file containing book data with columns id, title, description, and optionally ai_description.
- `<output_csv>`: Path to the output CSV file for categorized data.
- `<categories_file>`: Path to the text file containing a list of categories.

Example

```bash
python categorizer.py books.csv categorized_books.csv categories.txt
```

## Script Overview

The script performs the following tasks:

1. **Load Categories**: Loads categories from a text file.
2. **Process Books**:
   - Reads the book data from the input CSV.
   - Resumes from the last processed book if the output CSV already has data.
3. **Preprocess Descriptions**:
   - Cleans HTML tags from descriptions.
   - Combines titles and descriptions, truncating lengthy text to a manageable length.
4. **Category Selection**:
   - Uses OpenAI API to determine the most relevant categories from the provided list.
5. **Save Results**: Outputs the categorized data into a CSV file.

## Error Handling

- Rate Limiting: Implements exponential backoff if OpenAI API rate limits are exceeded.
- Data Validation: Ensures each book has a valid description before processing.
- Logging: Logs the processing status, including any errors or retries due to API rate limits.

## Logging

Logs are set to INFO level to monitor the script’s progress. This includes information on skipped books, retries, and general processing updates.

## License

This project is licensed under the MIT License.
