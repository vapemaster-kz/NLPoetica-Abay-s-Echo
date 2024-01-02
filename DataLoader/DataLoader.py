import os
from typing import List
import re

from loguru import logger

class DataLoader():
    def __init__(self, dir_path: str) -> None:
        if not os.path.isdir(dir_path):
            raise ValueError("Dataset folder is not found. Please provide correct path!")
        
        number_of_txt_files = self.count_txt_files_in_folder(dir_path)
        self.dir_path = dir_path

        logger.info(f"Successfully identifed {number_of_txt_files} of txt files in the dir folder")

    def count_txt_files_in_folder(self, folder_path: str) -> int:
        txt_files_count = 0

        for _, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.lower().endswith('.txt'):
                    txt_files_count += 1

        return txt_files_count

    def read_texts_from_folder(self, add_title: bool) -> List[str]:
        texts = []

        for root, _, files in os.walk(self.dir_path):
            for file_name in files:
                if file_name.lower().endswith('.txt'):
                    file_path = os.path.join(root, file_name)
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_text = file.read()
                        file_text = self.clean_text(file_text)
                        if add_title:
                            title = file_name[:-4]
                            file_text = f"{title}\n\n\n{file_text}"
                        
                        texts.append(file_text)

        return texts

    def clean_text(self, text:str) -> str:
        characters_to_remove = ['¹', '1', '2']

        pattern = '[' + re.escape(''.join(characters_to_remove)) + ']'

        result = re.sub(pattern, '', text)

        return result

