import csv
import sys


class LabelUtil:
    def __init__(self):
        self.filename = 'yamnet_class_map.csv'
        self.label_code_to_name = dict()
        self.label_name_to_code = dict()
        row_num = 0
        with open(self.filename, newline='') as f:
            reader = csv.reader(f, quotechar='"', delimiter=',',
                                quoting=csv.QUOTE_ALL, skipinitialspace=True)
            try:
                for row in reader:
                    if row_num > 0:
                        self.label_code_to_name[row[1]] = row[2]
                        self.label_name_to_code[row[2]] = row[1]
                    row_num += 1

            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(self.filename, reader.line_num, e))

    def get_code_names(self, codes):
        names = []
        for code in codes:
            names.append(self.label_code_to_name.get(code))

        return names

    def get_code_name(self, code):
        return self.label_code_to_name.get(code)

    def get_codes(self, names):
        codes = []
        for name in names:
            codes.append(self.label_name_to_code.get(name))

        return codes

    def get_code(self, name):
        return self.label_name_to_code.get(name)

# download_label_list = ['Dog',
#                        'Howl',
#                        'Cat',
#                        'Horse',
#                        'Pig',
#                        'Goat',
#                        'Sheep',
#                        'Fowl',
#                        'Turkey',
#                        'Mouse',
#                        'Frog',
#                        'Owl']
#
# download_code_list = ['/m/0bt9lr', '/m/07qf0zm', '/m/01yrx', '/m/03k3r', '/m/068zj', '/m/03fwl', '/m/07bgp', '/m/025rv6n', '/m/01rd7k', '/m/09d5_', '/m/04rmv', '/m/09ld4']
#
# labels = LabelUtil()
# print(labels.get_codes(download_label_list))
# print(labels.get_code_names(download_code_list))
