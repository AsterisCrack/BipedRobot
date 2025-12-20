import yaml
from config.schema import Config as PydanticConfig

class NoConfig:
    def __init__(self, config=None):
        pass

    def __getitem__(self, key):
        return NoConfig()
    
    def get(self, key, default=None):
        return default

    def __getattr__(self, name):
        return NoConfig()

    def __bool__(self):
        return False
    
class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self._config_data = self.load_config()
        # Convert scientific notation strings to numbers before validation if necessary
        self._config_data = self._convert_scientific_notation(self._config_data)
        # Validate with Pydantic
        self.config_obj = PydanticConfig(**self._config_data)

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file) or {}
    
    def copy_in_file(self, config_path):
        with open(config_path, 'w') as file:
            # Dump the Pydantic model back to dict/yaml
            yaml.dump(self.config_obj.model_dump(), file, default_flow_style=False)
            
    def __getitem__(self, key):
        # Allow dict-style access for backward compatibility
        val = getattr(self.config_obj, key)
        if hasattr(val, "model_dump"):
            return val.model_dump()
        return val

    def __getattr__(self, name):
        # Allow attribute access
        if name == "config_obj":
            raise AttributeError("config_obj not initialized")
        if not hasattr(self, "config_obj"):
             raise AttributeError(f"'{type(self).__name__}' object has no attribute 'config_obj'")
        return getattr(self.config_obj, name)
    
    def _convert_scientific_notation(self, obj):
        """
        Recursively explore the config and convert strings in scientific notation
        (e.g., '1e5', '-2e-3') into integers or floats.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._convert_scientific_notation(value)
        elif isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = self._convert_scientific_notation(obj[i])
        elif isinstance(obj, str):
            try:
                # Try to convert to float
                val = float(obj)
                # If it's an integer value, convert to int
                if val.is_integer():
                     return int(val)
                return val
            except:
                pass
        return obj
    
