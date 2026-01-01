import re, os, unicodedata
from config import PROJECT_ROOT, greekwords, greek_transliteration

class Preprocessor:
    def __init__(self):
        self.processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "meditations.txt")
        self.preprocessed_path = os.path.join(PROJECT_ROOT, "data", "raw", "meditations.txt")
        self.main()

    def preprocess(self, string):
        meditations = unicodedata.normalize("NFC", string)

        startindex = meditations.index("THE FIRST BOOK")
        meditations = meditations[startindex:]
        endindex = meditations.index("APPENDIX")
        meditations = meditations[:endindex] 

        book_name = r"THE\s+[A-Z]+\s+BOOK\s+[IVXLCDM]+\.\s"
        section_name = r"\n\n[IVXLCDM]+\. "
        underline = r"[_]+"
        book_end = r"\n\n\n\n"


        meditations = re.sub(book_name, "<BEGIN>\n", meditations)
        meditations = re.sub(book_end, "<END> ", meditations)
        meditations = re.sub(section_name, "\n<END>\n<BEGIN>\n", meditations)
        meditations = re.sub(underline, "", meditations)

        for i in range(len(greekwords)):
            meditations = meditations.replace(greekwords[i], greek_transliteration[i])

        return meditations

    def main(self):
        with open(self.preprocessed_path, "r") as f:
                self.meditations = f.read()

        self.meditations = self.preprocess(self.meditations)

        with open(self.processed_path, "w") as f:
            f.write(self.meditations)
            print("Successfully wrote Meditations.")
