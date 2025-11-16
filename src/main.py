from preprocess import load_corpus
from ngram_model import NGramModel
from markov_model import MarkovModel
from predictor import load_user_words, add_user_word

CORPUS_PATH = "data/corpus.txt"

def main():
    print("Loading corpus...")
    tokens = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(tokens)} tokens")

    print("Training models...")
    bigram = NGramModel(1)
    trigram = NGramModel(2)
    markov = MarkovModel()

    bigram.train(tokens)
    trigram.train(tokens)
    markov.train(tokens)

    while True:
        print("\n--- Predictive Text Generator ---")
        print("1. Predict next word (Trigram)")
        print("2. Predict next word (Bigram)")
        print("3. Predict next word (Markov)")
        print("4. Add custom word")
        print("5. Exit")
        
        option = input("Choose option: ").strip()
        
        if option == "1":
            text = input("Enter text: ").strip()
            pred = trigram.predict(text)
            print(f"Prediction: {pred}")
        elif option == "2":
            text = input("Enter text: ").strip()
            pred = bigram.predict(text)
            print(f"Prediction: {pred}")
        elif option == "3":
            text = input("Enter a word: ").strip()
            pred = markov.predict(text)
            print(f"Prediction: {pred}")
        elif option == "4":
            word = input("Enter word to add: ").strip()
            add_user_word(word)
            print(f"Added: {word}")
        elif option == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()