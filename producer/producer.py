import os
import base64
from pathlib import Path
from confluent_kafka import Producer
import logging
from dotenv import load_dotenv
import numpy as np
import signal
import time
import json
from typing import Dict, Optional, Any
from datetime import datetime, timezone, timedelta
from jsonschema import ValidationError, validate, FormatChecker
from syntheticgenerator import SyntheticTrafficGenerator

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


logger = create_logger('producer', 'producer.log')

SCHEMA = {
    'timestamp': {'type': 'time'},
    "event_id": {"type": "string"},
    "properties": {
    'src_port': {'type': 'int', 'range': (1024, 65535)},
    'dst_port': {'type': 'int', 'range': (1, 65535)},
    'protocol': {'type': 'cat', 'vals': ['TCP', 'UDP', 'ICMP']},
    'packet_count': {'type': 'int', 'range': (1, 2000)},
    'conn_state': {'type': 'cat', 'vals': ['EST', 'SYN', 'FIN', 'RST']},
    'bytes_transferred': {'type': 'float', 'range': (0.0, 2e6)},
    'label': {'type': 'int'}
}
}

class KafkaProducer:
    def __init__(self, seed: int = 42, start_ts: Optional[datetime]=None):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_username = os.getenv('KAFKA_USERNAME')
        self.kafka_password = os.getenv('KAFKA_PASSWORD')
        # Optional Kafka envs
        self.kafka_sasl_mechanism = os.getenv('KAFKA_SASL_MECHANISM', os.getenv('KAFKA_SASLMECHANISM', 'PLAIN'))
        self.kafka_security_protocol = os.getenv('KAFKA_SECURITY_PROTOCOL', os.getenv('KAFKA_SECURITYPROTOCOL', 'PLAINTEXT'))
        # Truststore: can be provided as a filesystem path or as a base64 encoded PEM
        self.topic = os.getenv('KAFKA_TOPIC', 'topic-traffic')
        self.running = False
        self.rng = np.random.RandomState(seed)
        self.generator = SyntheticTrafficGenerator(start_ts= start_ts, epsilon_seconds= 100, rng=self.rng)

        # confluent kafka config
        self.producer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': 'traffic_producer',
            'compression.type':'gzip',
            'linger.ms':'5',
            'batch.size': 16384
        }

        # Apply security settings based on environment
        if self.kafka_username and self.kafka_password:
            # Use provided security.protocol if set, otherwise default to SASL_SSL
            self.producer_config['security.protocol'] = self.kafka_security_protocol or 'SASL_SSL'
            # SASL mechanism from env (e.g. PLAIN, SCRAM-SHA-512)
            self.producer_config['sasl.mechanism'] = self.kafka_sasl_mechanism
            self.producer_config['sasl.username'] = self.kafka_username
            self.producer_config['sasl.password'] = self.kafka_password

            # If using SSL-based protocol and a truststore is provided as BASE64, write it to a temp file
            if 'SSL' in self.producer_config['security.protocol']:
                trust_path = None
                if self.kafka_truststore_path:
                    trust_path = self.kafka_truststore_path
                elif self.kafka_truststore_base64:
                    # decode to a temp file
                    trust_path = '/tmp/kafka_truststore.pem'
                    try:
                        decoded = base64.b64decode(self.kafka_truststore_base64)
                        Path(trust_path).write_bytes(decoded)
                    except Exception as e:
                        logger.error(f'Failed to write truststore file: {e}')
                        trust_path = None

                if trust_path:
                    # confluent-kafka uses ssl.ca.location for CA cert file
                    self.producer_config['ssl.ca.location'] = trust_path
        else:
            self.producer_config['security.protocol'] = self.kafka_security_protocol or 'PLAINTEXT'
        
        try:
            self.producer = Producer(self.producer_config)
            logger.info('Confluent Kafka Producer initialized successfully')
        except Exception as e:
            logger.error(f'Failed to initialize confluent kafka producer: {str(e)}')
            raise e
        
        # Configure graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')
    
    def validate_transaction(self, transaction: Dict[str, Any])->bool:
        try:
            validate(
                instance=transaction,
                schema=SCHEMA,
                format_checker = FormatChecker()
            )
            return True
        except ValidationError as e:
            logger.error(f'Invalid transaction: {e.message}')
    
    def send_transaction(self, trend:str) -> bool:
        try:
            transaction = self.generator.produce(trend=trend)
            if not transaction:
                return False
            
            self.producer.produce(
                self.topic,
                key=transaction['event_id'],
                value=json.dumps(transaction),
                callback=self.delivery_report
            )

            self.producer.poll(0) # trigger callbacks
            return True
        except Exception as e:
            logger.error(f'Error producing message: {str(e)}')

    def run_continuous_production(self, trend:str , interval: float=0.0):
        """Run continuous message production with graceful shutdown"""
        self.running = True
        logger.info('Starting producer for topic %s...', self.topic)
        try:
            while self.running:
                if self.send_transaction(trend):
                    time.sleep(interval)
        finally:
            self.shutdown()

    def shutdown(self, signum=None, frame=None): 
        if self.running:
            logger.info('Initiating shutdown...')
            self.running = False

            if self.producer:
                self.producer.flush(timeout=30)
                self.producer.close()
            logger.info('Producer stopped')

if __name__ == "__main__":
    producer = KafkaProducer()
    producer.run_continuous_production(trend='normal')