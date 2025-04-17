import yaml

class ConfigSubdict:
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, key):
        if key not in self.data:
            return None
        if type(self.data[key]) == dict:
            return ConfigSubdict(self.data[key])
        return self.data[key]
    
class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self._convert_scientific_notation(self.config)

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def copy_in_file(self, config_path):
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
            
    def __getitem__(self, key):
        if key not in self.config:
            return None
        if type(self.config[key]) == dict:
            return ConfigSubdict(self.config[key])
        return self.config[key]
    
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
                return float(obj)
            except:
                pass
        return obj