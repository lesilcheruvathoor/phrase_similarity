import logging
import sys
import pandas as pd
from utils import *
from data_process import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    try:
        print("Starting")
        # Load the phrases
        phrase_df = load_phrases('phrases.csv')
        # Load the vectors
        word_vect = file_generated_for_vectors()
        # Assign vectors to phrases
        phrase_df["vector"] = phrase_df["Phrases"].apply(lambda x: get_phrase_vector(x, word_vect))
        # Calculate the distance
        wd_dist = calc_distances(phrase_df)
        # Save the word distance
        save_file(wd_dist,'word_distances.csv')

        # Read the input text file for use case
        ip_phrase = read_input_file()
        new_phrase, distance = identify_phrase(ip_phrase, word_vect, phrase_df)

        logger.info(f"The closest phrase is '{new_phrase}' and the distance is {distance}")
    except Exception as e:
        logger.error(f"Error occured: {e}")


if __name__ == '__main__':
    main()
