import sys
from P1_Bpe.bpe import ByteLevelBPE, encode_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
            print("Uso:")
            print("  python p1_bpe.py train <input_train_corpus> <output_model_file>")
            print("  python p1_bpe.py eval <input_model_file> <input_text>")
            sys.exit(1)

    mode = sys.argv[1]
    bpe = ByteLevelBPE()

    if mode == "train":
        if len(sys.argv) != 4:
            print("Uso: python p1_bpe.py train <input_train_corpus> <output_model_file>")
            sys.exit(1)
        input_corpus = sys.argv[2]
        output_model = sys.argv[3]
        with open(input_corpus, "r", encoding="utf-8") as f:
            lines = f.readlines()
        bpe.train(lines)
        
        # Test de verificación después del entrenamiento
        test_text = "Hola mundo"
        print(f"\n--- Test de verificación ---")
        print(f"Texto de prueba: '{test_text}'")
        
        encoded = bpe.encode(test_text)
        decoded = bpe.decode(encoded)
        tokens = bpe.tokenize(test_text)
        
        print(f"Encoded: {encoded}", end="\n\n")
        print(f"Tokens: {tokens}", end="\n\n")
        print(f"Decoded: '{decoded}'", end="\n\n")
        print(f"¿Codificación correcta?: {test_text == decoded}", end="\n\n")
        
        bpe.save(output_model)
        print(f"Modelo guardado en: {output_model}", end="\n\n")

    elif mode == "eval":
        if len(sys.argv) < 4 or len(sys.argv) > 5:
            print("Uso: python p1_bpe.py eval <input_model_file> <input_text> [-f]")
            print("  -f: indica que <input_text> es un archivo")
            sys.exit(1)
        
        input_model = sys.argv[2]
        input_text_or_file = sys.argv[3]
        
        # Check if -f flag is present
        if len(sys.argv) == 5 and sys.argv[4] == "-f":
            # Read from file
            with open(input_text_or_file, "r", encoding="utf-8") as f:
                input_text = f.read()
        else:
            # Use as direct text
            input_text = input_text_or_file
        
        bpe.load(input_model)
        
        # Mostrar diferentes representaciones
        token_ids = bpe.encode(input_text)
        tokens = bpe.tokenize(input_text)
        decoded_text = bpe.decode(token_ids)
        
        print("Texto original:", repr(input_text), end="\n\n")
        print("Token IDs:", token_ids, end="\n\n")
        print("Tokens (string):", tokens, end="\n\n")
        print("Texto decodificado:", repr(decoded_text), end="\n\n")
        print("¿Decodificación correcta?:", input_text == decoded_text, end="\n\n")

    elif mode == "tokenize_save":
        if len(sys.argv) != 5:
            print("Uso: python p1_bpe.py tokenize_save <input_model_file> <input_text_file> <output_tokens_file>")
            sys.exit(1)
        
        input_model = sys.argv[2]
        input_text_file = sys.argv[3]
        output_tokens_file = sys.argv[4]
        
        encoded = encode_file(input_model, input_text_file)
        with open(output_tokens_file, "w", encoding="utf-8") as f:
            f.write(' '.join(map(str, encoded)))
        print(f"Tokens guardados en: {output_tokens_file}", end="\n\n")

    else:
        print("Modo no reconocido. Usa `train` o `eval`.")