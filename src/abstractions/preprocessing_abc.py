from abc import ABC, abstractmethod
from pathlib import Path

class AbstractPreprocess(ABC):
    @abstractmethod
    def load_docs(self, path: Path):
        pass
    
    @abstractmethod
    def text_normalize(self, docs):
        pass