#!/usr/bin/env bash



# MALAY RUN

MALAY_CSV_PATH="data/malay/malay_12000.csv"
MALAY_OUT_DIR="gen/malay"

python3 -m scripts.generate_malay -in ${MALAY_CSV_PATH} -out ${MALAY_OUT_DIR} -n malay


# WELSH RUN

WELSH_CSV_PATH="data/welsh/welsh_12000.csv"
WELSH_OUT_DIR="gen/welsh"

python3 -m scripts.generate_welsh -in ${WELSH_CSV_PATH} -out ${WELSH_OUT_DIR} -n welsh


# FRENCH RUN

FRENCH_CSV_PATH="data/french/french_4500.csv"
FRENCH_OUT_DIR="gen/french"

python3 -m scripts.generate_french -in ${FRENCH_CSV_PATH} -out ${FRENCH_OUT_DIR} -n french


# TELUGU RUN

TELUGU_CSV_PATH="data/telugu/telugu_12000.csv"
TELUGU_OUT_DIR="gen/telugu"

# python3 -m scripts.generate_telugu -in ${TELUGU_CSV_PATH} -out ${TELUGU_OUT_DIR} -l 1 -n telugu


# AVAR RUN

AVAR_CSV_PATH="data/avar/avar_12000.csv"
AVAR_OUT_DIR="gen/avar"

python3 -m scripts.generate_avar -in ${AVAR_CSV_PATH} -out ${AVAR_OUT_DIR} -n avar
