import os
from pypdf import PdfReader

def extract_text_from_pdfs(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    for filename in files:
        path = os.path.join(directory, filename)
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            output_filename = f"{filename}.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted {filename} to {output_filename}")
        except Exception as e:
            print(f"Failed to extract {filename}: {e}")

if __name__ == "__main__":
    extract_text_from_pdfs("d:/AI/interview_project/file")
