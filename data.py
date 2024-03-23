
import os



class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path # dir or file path
        self.data = ""

    def load_data(self):
        if os.path.isdir(self.data_path):
            for file in os.listdir(self.data_path):
                with open(os.path.join(self.data_path, file), 'r', encoding="utf-8-sig") as f:
                    self.data += f.read().strip('\n')
        else:
            with open(self.data_path, 'r', encoding="utf-8-sig") as f:
                self.data = f.read().strip('\n')
    
    def get_splits(self, data, train_size=0.9) -> tuple:
        # data may be encoded or not
        train_size = int(len(data) * train_size)
        train = data[:train_size]
        val = data[train_size:]
        return train, val


            