# Conflict-Insight-Bot

ConflictInsightBot is a Python script designed to analyze and summarize war-related articles, particularly focusing on the Israel-Hamas conflict. The script uses BM25 for retrieving relevant articles and a pre-trained QA model for extracting answers based on user queries. It also integrates Wikipedia to provide additional context.

## Features

* <b>Data Preprocessing:</b> Loads and preprocesses articles from a JSON file, filtering for war-related content.

* <b>Article Retrieval:</b> Uses the BM25 algorithm to retrieve the most relevant articles based on user queries.

* <b>Question Answering:</b> Utilizes a pre-trained QA model to extract answers from the retrieved articles.

* <b>Wikipedia Integration:</b> Augments answers with summaries from Wikipedia for comprehensive information.

* <b>Summarization:</b> Combines answers from articles and Wikipedia into a formatted summary.

* <b>Logging and Error Handling:</b> Includes logging for process tracking and error handling for robustness.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Python 3.11.7
* Internet connection (for downloading models and Wikipedia summaries)
* Required Python packages:

    * `json`
    * `re`
    * `logging`
    * `rank_bm25`
    * `transformers`
    * `wikipedia`
    * `warnings`
    * `nltk`

## Installation

1. Clone the Repository

```bash
git clone https://github.com/yourusername/ ConflictInsightBot.git
cd ConflictInsightBot 
```

2. Install Dependencies 

```bash
pip install -r requirements.txt
```

3. Download NLTK Dependencies

```bash
import nltk
nltk.download('punkt')
```

## Usage

1. <b>Prepare the Data</b>

Ensure you have a JSON file containing the articles you want to analyze. Update the file path in the script to point to your JSON file:

```bash
file_path = '/path/to/your/news.article.json'
```

2. <b>Run the script</b>

```bash
python name_of_your_python_script.py 
```

3. <b>Interact with the Bot</b>

```bash
Enter your question (or type 'exit' to quit):
```
</br>

---
Sample questions that a user might ask the ConflictInsightBot:

* What happened at Al-Shifa Hospital? 
* What is the current state of the Israel-Hamas conflict?
* What led to the outbreak of the latest conflict between Israel and Hamas?
* What are the reported casualties and damages from the recent clashes?
* Are there any ongoing peace talks between Israel and Palestine?
* How is the media covering the Israel-Hamas conflict?
* How has the conflict affected the economy of Gaza?
* What are the geopolitical interests of regional powers in the Israel-Palestine conflict?
* What measures can be taken to prevent future escalations?
---
</br>

4. <b>View the results</b>

</br>
The script will display the summarized answers from the articles and Wikipedia. It will also save the top relevant articles to a specified file.

## Script Breakdown

* <b>load_and_preprocess_data(file_path):</b> Loads and preprocesses articles from a JSON file, filtering for war-related content.

* <b>retrieve_relevant_articles(query, articles):</b> Retrieves the most relevant articles using the BM25 algorithm.

* <b> load_qa_model(model_name):</b> Loads a pre-trained QA model.

* <b>get_answer_from_articles(query, articles, qa_pipeline):</b> Extracts answers from articles using the QA model.

* <b>augment_with_wikipedia(query):</b> Retrieves a summary from Wikipedia.

* <b>summarize_answers(query, article_answers, wiki_summary):</b> Combines answers from articles and Wikipedia into a formatted summary.

* <b>save_articles_to_file(articles, file_path):</b> Saves articles to a specified file. In this case, `top_articles.txt`

* <b>main():</b> The main function to run the script, handle user input, and display results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/abhilashaojha/Conflict-Insight-Bot/blob/main/LICENSE) file for details.

## Acknowledgments

* The [rank_bm25](https://github.com/dorianbrown/rank_bm25) library by Dorian Brown for the BM25 implementation.
* The [transformers](https://github.com/huggingface/transformers) library for providing the pre-trained QA model.
* The [wikipedia](https://github.com/goldsmith/Wikipedia) library for easy access to Wikipedia summaries.
* [NLTK](https://www.nltk.org/) for natural language processing utilities.



