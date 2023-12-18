from box import Box
from abc import ABC, abstractmethod

class AbstractPipeline(ABC):

    @abstractmethod
    def run(self, context = Box({})):
        pass
