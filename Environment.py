
class Environment:
    @staticmethod
    def read_token():
        with open("token.txt", "r") as file:
            return file.read()