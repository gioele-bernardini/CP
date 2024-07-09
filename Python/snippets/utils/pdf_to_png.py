#! /usr/bin/env python3

import fitz  # PyMuPDF
from PIL import Image

def convert_pdf_to_png(pdf_path, output_folder):
  # Apri il PDF
  pdf_document = fitz.open(pdf_path)
  for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)

    # Rendi la pagina un'immagine
    pix = page.get_pixmap()
    # Crea il percorso del file per l'immagine
    output_path = f"{output_folder}/page_{page_num + 1}.png"
    # Salva l'immagine in formato PNG
    pix.save(output_path)
    print(f"Pagina {page_num + 1} salvata come {output_path}")

# Esempio di utilizzo
pdf_path = "/home/gioele/Documents/iscrizione.pdf"
output_folder = "."
convert_pdf_to_png(pdf_path, output_folder)

