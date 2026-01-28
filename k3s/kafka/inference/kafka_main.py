import os
import sys
import signal
import logging
from typing import Dict, Optional
import json
import ray
import httpx
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException


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

class KafkaConfig:
    """Kafka configuration from environment variables."""
    
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
        self.username = os.getenv('KAFKA_USERNAME')
        self.password = os.getenv('KAFKA_PASSWORD')
        self.sasl_mechanism = os.getenv('KAFKA_SASL_MECHANISM', 'PLAIN')
        self.security_protocol = os.getenv('KAFKA_SECURITY_PROTOCOL', 'SASL_SSL')
        self.ssl_ca_location = os.getenv('KAFKA_SSL_CA_LOCATION')
        self.input_topic = os.getenv('KAFKA_TOPIC', 'topic-traffic')
        self.output_topic = os.getenv('KAFKA_TOPIC_OUTPUT', 'topic-prediction')
        self.group_id = 'kafka-inference-consumer-group'

        # Ray Serve HTTP endpoint (recommended when Serve runs in a different Ray cluster)
        # Example in-cluster: http://model-serving-serve-svc.ray.svc.cluster.local:8000/infer
        self.ray_serve_url = os.getenv('RAY_SERVE_URL')
        self.ray_serve_timeout_s = float(os.getenv('RAY_SERVE_TIMEOUT_S', '10'))
        
        self._validate()
    
    def _validate(self):
        """Validate required configuration."""
        required = [
            self.bootstrap_servers,
            self.username,
            self.password,
            self.input_topic,
            self.output_topic
        ]
        if not all(required):
            raise ValueError("Missing required Kafka configuration")


@ray.remote
class KafkaInferenceActor:
    """
    Ray actor that consumes from Kafka, calls Ray Serve for prediction,
    and produces results back to Kafka with exactly-once semantics.
    """
    
    def __init__(self, config: KafkaConfig, actor_id: int):
        self.config = config
        self.actor_id = actor_id
        self.logger = logging.getLogger(f'KafkaInferenceActor-{actor_id}')
        self.running = True
        
        self.consumer = None
        self.producer = None
        self._http: httpx.Client | None = None
        
        self._initialize_kafka()
        self._initialize_predictor()
        
        self.logger.info(f"Actor {actor_id} initialized successfully")
    
    def _initialize_kafka(self):
        """Initialize Kafka consumer and producer with exactly-once semantics."""
        try:
            consumer_config = {
                'bootstrap.servers': self.config.bootstrap_servers,
                'sasl.mechanism': self.config.sasl_mechanism,
                'security.protocol': self.config.security_protocol,
                'sasl.username': self.config.username,
                'sasl.password': self.config.password,
                'group.id': self.config.group_id,
                'auto.offset.reset': 'earliest',
                'enable.auto.commit': False,
                'isolation.level': 'read_committed',
                'max.poll.interval.ms': 300000,
                'session.timeout.ms': 45000
            }

            if self.config.ssl_ca_location:
                consumer_config['ssl.ca.location'] = self.config.ssl_ca_location
            
            producer_config = {
                'bootstrap.servers': self.config.bootstrap_servers,
                'sasl.mechanism': self.config.sasl_mechanism,
                'security.protocol': self.config.security_protocol,
                'sasl.username': self.config.username,
                'sasl.password': self.config.password,
                'enable.idempotence': True,
                'acks': 'all',
                'max.in.flight.requests.per.connection': 5,
                'compression.type': 'lz4'
            }

            if self.config.ssl_ca_location:
                producer_config['ssl.ca.location'] = self.config.ssl_ca_location
            
            self.consumer = Consumer(consumer_config)
            self.producer = Producer(producer_config)
            
            self.consumer.subscribe([self.config.input_topic])
            self.logger.info(f"Actor {self.actor_id} subscribed to topic: {self.config.input_topic}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka: {e}")
            raise
    
    def _initialize_predictor(self):
        """Initialize predictor client.

        This consumer is designed to run in a Ray cluster that may be different from the
        Ray Serve cluster that exposes the HTTP route (e.g. /infer). For that reason,
        we default to HTTP calls instead of `serve.get_deployment(...).get_handle()`.
        """
        try:
            if not self.config.ray_serve_url:
                raise ValueError(
                    "Missing RAY_SERVE_URL (e.g. http://model-serving-serve-svc.ray.svc.cluster.local:8000/infer)"
                )

            self._http = httpx.Client(timeout=self.config.ray_serve_timeout_s)

            # Optional quick health check
            healthz = self.config.ray_serve_url.rstrip('/') + '/healthz'
            try:
                r = self._http.get(healthz)
                if r.status_code >= 400:
                    self.logger.warning(
                        f"Ray Serve health check non-200: {r.status_code} body={r.text[:200]}"
                    )
            except Exception as e:
                self.logger.warning(f"Ray Serve health check failed: {e}")

            self.logger.info(
                f"Actor {self.actor_id} configured to call Ray Serve via HTTP: {self.config.ray_serve_url}"
            )
        except Exception as e:
            self.logger.error(f"Failed to get predictor handle: {e}")
            raise
    
    def read_from_kafka(self) -> Optional[Dict]:
        """
        Read a single message from Kafka.
        
        Returns:
            Dict with 'data', 'key', 'partition', 'offset' or None if no message
        """
        try:
            msg = self.consumer.poll(timeout=1.0)
            
            if msg is None:
                return None
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    return None
                else:
                    raise KafkaException(msg.error())
            
            data = json.loads(msg.value().decode('utf-8'))
            
            return {
                'data': data,
                'key': msg.key().decode('utf-8') if msg.key() else None,
                'partition': msg.partition(),
                'offset': msg.offset()
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode message: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading from Kafka: {e}")
            raise
    
    def predict(self, data: Dict) -> Dict:
        """
        Call Ray Serve predictor.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result
        """
        try:
            if self._http is None:
                raise RuntimeError("Predictor client not initialized")

            payload = {"data": data}
            resp = self._http.post(self.config.ray_serve_url, json=payload)
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result.get("error")))
            return result
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def stream_to_kafka(self, result: Dict, key: Optional[str] = None):
        """
        Produce prediction result to Kafka output topic.
        
        Args:
            result: Prediction result to write
            key: Optional message key
        """
        try:
            value = json.dumps(result).encode('utf-8')
            key_bytes = key.encode('utf-8') if key else None
            
            self.producer.produce(
                topic=self.config.output_topic,
                value=value,
                key=key_bytes,
                on_delivery=self._delivery_callback
            )
            
            self.producer.poll(0)
            
        except Exception as e:
            self.logger.error(f"Failed to produce to Kafka: {e}")
            raise
    
    def _delivery_callback(self, err, msg):
        """Callback for producer delivery reports."""
        if err:
            self.logger.error(f"Message delivery failed: {err}")
    
    def _commit_offset(self):
        """Commit current offset to ensure exactly-once processing."""
        try:
            self.consumer.commit(asynchronous=False)
        except Exception as e:
            self.logger.error(f"Failed to commit offset: {e}")
            raise
    
    def process_message(self, message: Dict):
        """
        Process a single message: predict and produce.
        
        Args:
            message: Message dict from read_from_kafka
        """
        try:
            self.logger.info(f"Actor {self.actor_id} processing partition {message['partition']} offset {message['offset']}")
            
            prediction = self.predict(message['data'])
            
            self.stream_to_kafka(
                result=prediction,
                key=message['key']
            )
            
            self.producer.flush(timeout=10)
            
            self._commit_offset()
            
            self.logger.info(f"Actor {self.actor_id} completed offset {message['offset']}")
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            raise
    
    def run(self):
        """Main processing loop."""
        self.logger.info(f"Actor {self.actor_id} starting consumer loop")
        
        try:
            while self.running:
                message = self.read_from_kafka()
                
                if message is None:
                    continue
                
                self.process_message(message)
                
        except KeyboardInterrupt:
            self.logger.info(f"Actor {self.actor_id} received interrupt signal")
        except Exception as e:
            self.logger.error(f"Actor {self.actor_id} fatal error: {e}")
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of Kafka connections."""
        self.logger.info(f"Actor {self.actor_id} shutting down")
        self.running = False
        
        try:
            if self.producer:
                self.producer.flush(timeout=10)
                self.logger.info(f"Actor {self.actor_id} producer flushed")
            
            if self.consumer:
                self.consumer.close()
                self.logger.info(f"Actor {self.actor_id} consumer closed")
                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def stop(self):
        """Signal actor to stop processing."""
        self.running = False


def main():
    """Entry point for RayJob."""
    logger = logging.getLogger('kafka_consumer_main')
    
    try:
        logger.info("Starting Kafka inference consumer")

        if not ray.is_initialized():
            # In RayJob contexts, connect to the in-cluster Ray runtime.
            ray.init(address="auto")
        
        config = KafkaConfig()
        num_actors = int(os.getenv('NUM_ACTORS', '6'))
        
        logger.info(f"Creating {num_actors} consumer actors")
        actors = [
            KafkaInferenceActor.remote(config, actor_id=i) 
            for i in range(num_actors)
        ]
        
        logger.info("Starting all actors")
        futures = [actor.run.remote() for actor in actors]
        
        logger.info("Waiting for actors to complete")
        ray.get(futures)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()