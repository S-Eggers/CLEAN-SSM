from abc import ABC, abstractmethod


class Preparation(ABC):    
    @abstractmethod
    def run(self):
        pass
