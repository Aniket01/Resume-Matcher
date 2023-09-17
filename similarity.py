from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class Similarity:
    """
    A Class for obtaining cosine similarity between two sentence embeddings
    """

    def __init__(self, txt1: str, txt2: str):
        """ 
        Initializes the Similarity object

        Args:
            txt1 (str): input text from resume 
            txt2 (str): input text from job description
        """
        self.txt1 = txt1
        self.txt2 = txt2
        self.embeddings1 = model.encode(txt1)
        self.embeddings2 = model.encode(txt2)

    def calculate(self):
        """
        Calculate Cosine_similarity between two sentence embeddings
        using sklearn.metrics.pairwise.cosine_similarity()

        Returns:
            float in range (0-1)
        """
        self.embeddings1 = self.embeddings1.reshape(1,-1)
        self.embeddings2 = self.embeddings2.reshape(1,-1)
        return float(cosine_similarity(self.embeddings1, self.embeddings2))

