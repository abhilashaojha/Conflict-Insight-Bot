import json
import re
import logging
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import wikipedia
import warnings
import nltk

# Ensure nltk dependencies are downloaded
nltk.download('punkt')

# Suppress warnings and unnecessary logging
warnings.filterwarnings('ignore')
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the data from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the articles.

    Returns:
        list: A list of cleaned war-related articles.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            articles = json.load(file)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return []

    war_related_articles = []
    keywords = ["Israel", "Hamas", "Palestine", "Gaza", "war", "conflict"]
    for article in articles:
        content = article.get('articleBody', '')
        if any(keyword in content for keyword in keywords):
            clean_content = re.sub(r'[^\w\s]', '', content)
            war_related_articles.append(clean_content)

    logging.info(f"Loaded and preprocessed {len(war_related_articles)} articles.")
    return war_related_articles


def retrieve_relevant_articles(query, articles):
    """
    Retrieves the most relevant articles based on the query using the BM25 algorithm.

    Args:
        query (str): The search query.
        articles (list): A list of articles to search within.

    Returns:
        list: A list of the top N relevant articles.
    """
    tokenized_articles = [nltk.word_tokenize(article.lower()) for article in articles]
    bm25 = BM25Okapi(tokenized_articles)
    tokenized_query = nltk.word_tokenize(query.lower())
    top_n = bm25.get_top_n(tokenized_query, articles, n=10)
    logging.info(f"Retrieved {len(top_n)} relevant articles for the query '{query}'.")
    return top_n


def load_qa_model(model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
    """
    Loads the QA model and tokenizer.

    Args:
        model_name (str): The name of the pretrained QA model.

    Returns:
        Pipeline: A QA pipeline.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        logging.info(f"Loaded QA model '{model_name}'.")
        return qa_pipeline
    except Exception as e:
        logging.error(f"Error loading QA model: {e}")
        return None


def get_answer_from_articles(query, articles, qa_pipeline):
    """
    Retrieves answers from articles based on the query using the QA model.

    Args:
        query (str): The search query.
        articles (list): A list of articles to search within.
        qa_pipeline (Pipeline): The QA pipeline.

    Returns:
        list: A list of answers extracted from the articles.
    """
    pre_prompt = ("You are an investigative bot designed to analyze articles regarding the Israel-Hamas conflict. "
                  "Summarize the following content based on the user's question in 4-5 sentences to the best of your ability and knowledge.")
    full_query = pre_prompt + " " + query

    answers = []
    for article in articles:
        result = qa_pipeline(question=full_query, context=article)
        if result:
            answers.append(result['answer'])

    logging.info(f"Extracted answers from {len(answers)} articles.")
    return answers


def augment_with_wikipedia(query):
    """
    Augments the answer with additional information from Wikipedia.

    Args:
        query (str): The search query.

    Returns:
        str: A summary from Wikipedia.
    """
    try:
        summary = wikipedia.summary(query, sentences=5)
    except wikipedia.exceptions.DisambiguationError as e:
        summary = wikipedia.summary(e.options[0], sentences=5)
    except wikipedia.exceptions.PageError:
        summary = "No additional information found."
    # except Exception as e:
    #     logging.error(f"Error retrieving Wikipedia summary: {e}")
    #     summary = "Error retrieving information."
    return summary


def summarize_answers(query, article_answers, wiki_summary):
    """
    Summarizes and formats the answers from articles and Wikipedia.

    Args:
        query (str): The search query.
        article_answers (list): A list of answers from articles.
        wiki_summary (str): A summary from Wikipedia.

    Returns:
        str: A formatted summary.
    """
    combined_answers = ' '.join(article_answers)

    summary = (f"\nBased on the context of Israel-Hamas conflict and your question '{query}', here is a summary:\n"
               f"{combined_answers}\n\n"
               f"Additional information from Wikipedia:\n{wiki_summary}")
    return summary


def save_articles_to_file(articles, file_path):
    """
    Saves the articles to a file.

    Args:
        articles (list): A list of articles to save.
        file_path (str): The path to the file where articles will be saved.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for article in articles:
                file.write(article + "\n")
        logging.info(f"Saved {len(articles)} articles to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving articles to file: {e}")


def main():
    file_path = '/content/drive/MyDrive/news.article.json'  # Update this path
    articles = load_and_preprocess_data(file_path)
    qa_pipeline = load_qa_model()

    if not qa_pipeline:
        logging.error("Exiting due to QA model loading failure.")
        return

    accumulated_answers = []

    while True:
        query = input("\n\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit' or not query.strip():
            break

        relevant_articles = retrieve_relevant_articles(query, articles)
        save_articles_to_file(relevant_articles, '/content/drive/MyDrive/top_articles.txt')  # Update this path

        answers = get_answer_from_articles(query, relevant_articles, qa_pipeline)
        wiki_summary = augment_with_wikipedia(query)

        summary = summarize_answers(query, answers, wiki_summary)
        print(summary)

        accumulated_answers.append(summary)

    print("\nAccumulated Answers:")
    for answer in accumulated_answers:
        print(answer, "\n")


if __name__ == "__main__":
    main()
