import logging
import yaml

def create_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Crea y configura un logger con consola y archivo opcional.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Evitar duplicar handlers si la función se llama varias veces
    if not logger.handlers:
        # Handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler de archivo, solo si se pasa log_file
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.ERROR)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger



class BaseUtils:
    """
    Clase base con métodos utilitarios para cargar parámetros y usar logging.
    """
    def __init__(self, logger: logging.Logger, params_path: str):
        self.logger = logger
        self.params_path = params_path

    def load_params(self) -> dict:
        """
        Carga un archivo YAML y retorna un diccionario con los parámetros.
        """
        try:
            with open(self.params_path, 'r') as file:
                params = yaml.safe_load(file)
            self.logger.debug('Parameters retrieved from %s', self.params_path)
            return params
        except FileNotFoundError:
            self.logger.error('File not found: %s', self.params_path)
            raise
        except yaml.YAMLError as e:
            self.logger.error('YAML error: %s', e)
            raise
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise