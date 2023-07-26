import os


class FileHandler:
    @staticmethod
    def set_folder(folder_name):
        temp = ''
        folder_list = folder_name.split('/')
        for i in folder_list[1:]:
            temp += f'/{i}'
            if not os.path.exists(temp):
                os.mkdir(temp)
        return folder_name
