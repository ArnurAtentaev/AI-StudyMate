from abc import ABC, abstractmethod


class AbstractPreprocess(ABC):
    @abstractmethod
    def load_docs(self):
        pass
    
    @abstractmethod
    def text_normalize(self):
        pass