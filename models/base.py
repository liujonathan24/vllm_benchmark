from typing import List, Optional, Dict
import gc
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract Base Class for a model wrapper.
    It defines the required methods that all concrete
    model implementations must provide.
    """
    def __init__(self, model_name: str, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        if self.verbose:
            print(f"Base init for {self.model_name}")

    @abstractmethod
    def generate(self, prompt: str) -> Optional[str]:
        """
        Applies next token prediction (raw completion) on the input string.
        """ 
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Applies the model's chat template to a conversation history
        and returns the next response.
        """
        pass

    @abstractmethod
    def destroy(self):
        """
        Explicitly destroys the model and frees all associated GPU memory.
        """
        pass
        
    def __del__(self):
        """
        Fallback destructor, ensures destroy() is called when the
        object is garbage collected.
        """
        if self.verbose:
            print(f"Fallback __del__ triggered for {self.model_name}")
        self.destroy()