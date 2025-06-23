import tiktoken
from transformers import AutoTokenizer

class TokenCounter:
    def __init__(self):
        self.tokenizers = {}
    
    def get_tiktoken_count(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens using tiktoken (OpenAI's tokenizer)"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            # Fallback to cl100k_base encoding
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    
    def get_hf_tokenizer_count(self, text: str, model_name: str) -> int:
        """Count tokens using HuggingFace tokenizer"""
        try:
            if model_name not in self.tokenizers:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True
                )
            tokenizer = self.tokenizers[model_name]
            return len(tokenizer.encode(text))
        except Exception as e:
            print(f"Could not load tokenizer for {model_name}: {e}")
            return self._fallback_count(text)
    
    def _fallback_count(self, text: str) -> int:
        """Fallback token counting (words + punctuation)"""
        import re
        # More accurate than char/4 - counts words and punctuation separately
        words = len(re.findall(r'\w+', text))
        punctuation = len(re.findall(r'[^\w\s]', text))
        return words + punctuation 