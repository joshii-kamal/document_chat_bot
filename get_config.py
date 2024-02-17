import json
import os
# from get_cwd import get_project_directory

class Config:
    def __init__(self, filename="D:\COMPANY\plan_extract_simplified\config.json"):
        self.filename = filename
        self.config_data = self.get_config()

    def get_config(self):
        try:
            with open(self.filename, "r",encoding='UTF-8') as config_file:
                config = json.load(config_file)
            return config
        except FileNotFoundError:
            raise Exception(f"Config file '{self.filename}' not found.")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON in config file '{self.filename}'.")

    def __getitem__(self, key):
        return self.config_data.get(key)
