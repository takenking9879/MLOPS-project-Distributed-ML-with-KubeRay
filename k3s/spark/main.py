from utils import BaseUtils, create_logger
import boto3
import os
from pyspark.sql import SparkSession
import importlib
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType

class SparkPreprocessing(BaseUtils):
    def __init__(self, schema: StructType,  params_path: str, data_dir: str, output_dir: str, artifacts_dir: str):
        logger = create_logger('SparkPreprocessing', 'spark_preprocessing.log')
        super().__init__(logger, params_path)
        self.params = self.load_params()['spark']
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.artifacts_dir = artifacts_dir
        self.schema = schema
        self.spark = self._create_spark_session()
        self.scaler = None
        self.s3 = None

    def _check_minio_connection(self):
        try:
            s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-2'),
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            self.s3 = s3
            self.logger.info('Minio connection verified. Buckets: %s', bucket_names)
        except Exception as e:
            self.logger.error('Minio connection failed: %s', str(e), exc_info=True)
            raise

    @staticmethod
    def _to_s3a_path(path: str) -> str:
        if path.startswith('s3a://'):
            return path
        if path.startswith('s3://'):
            return 's3a://' + path[len('s3://'):]
        return path
        
    def _create_spark_session(self):
        try:
            self._check_minio_connection()
            self.logger.info("Creating SparkSession with S3A (MinIO) support")
            spark = (
            SparkSession.builder
            .appName(self.params['app_name'])
            .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
            .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))
            
            # 1. Especificar el proveedor de credenciales (evita confusiones internas de Hadoop)
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            .config("spark.hadoop.fs.s3a.path.style.access", "false")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            
            # 2. Rendimiento: Esto ayuda mucho con Parquets grandes en S3
            .config("spark.hadoop.fs.s3a.experimental.fadvise", "random")
            .config("spark.sql.files.maxPartitionBytes", self.params.get('read_batch_size', 256) * 1024 * 1024)
            
            # 3. Magic Committer (Ya lo tenías, ¡excelente!)
            .config("spark.hadoop.fs.s3a.committer.name", "magic")
            .config("spark.hadoop.fs.s3a.committer.magic.enabled", "true")
            .getOrCreate()
        )
        except Exception as e:
            self.logger.error('Cannot create Spark session: %s', str(e))
            raise

        return spark

    def load_data(self, file_path: str):
        try:

            spark = self.spark
            self.logger.info(f"Loading data from {file_path}")
            df = spark.read.schema(self.schema).parquet(file_path)
            self.logger.info(
                f"Data loaded successfully | partitions: {df.rdd.getNumPartitions()}"
            )
            return df

        except Exception as e:
            self.logger.error("Failed loading data", exc_info=True)
            raise

    def preprocess(self, dataset: str = 'train'):
        """
        Unified preprocessing entry.
        """
        try:
            if dataset not in ['train', 'val', 'test']:
                raise ValueError("dataset must be one of ['train', 'val', 'test']")

            pipeline_module = self.params['pipeline']['module']
            self.logger.info(f"Loading feature pipeline: {pipeline_module}")
            module = importlib.import_module(pipeline_module)

            if dataset == 'train':
                df = self.load_data(os.path.join(self.data_dir, f'{dataset}/'))
                df_out, pipeline_model = module.preprocess_spark(df, model=self.scaler, train=True)
                self.scaler = pipeline_model
                self._save_scaler_artifact()
            else:
                df = self.load_data(os.path.join(self.data_dir, f'{dataset}/'))
                if self.scaler is None:
                    self.scaler = self.load_scaler_artifact()
                df_out, _ = module.preprocess_spark(df, model=self.scaler, train=False)

            self.write_data(df_out, os.path.join(self.output_dir, f'{dataset}/'))
        except Exception as e:
            self.logger.error('Preprocess failed to complete: %s', str(e), exc_info=True)
            raise

    def _save_scaler_artifact(self):
        """Spark guarda modelos directamente en S3A."""
        try:
            s3a_path = self._to_s3a_path(self.artifacts_dir)
            model_path = os.path.join(s3a_path, 'pipeline_model')
            
            self.scaler.write().overwrite().save(model_path)
            self.logger.info(f'PipelineModel saved to {model_path}')
        except Exception as e:
            self.logger.error('Failed saving model to S3', exc_info=True)
            raise

    def load_scaler_artifact(self):
        """Carga el PipelineModel desde S3A."""
        try:
            s3a_path = self._to_s3a_path(self.artifacts_dir)
            model_path = os.path.join(s3a_path, 'pipeline_model')
            
            from pyspark.ml import PipelineModel
            self.scaler = PipelineModel.load(model_path)
            self.logger.info(f'PipelineModel loaded from {model_path}')
            return self.scaler
        except Exception as e:
            self.logger.error('Failed loading model from S3', exc_info=True)
            raise
    
    def write_data(self, df, output_path: str):
        """
        Escribe DataFrame en parquet en S3 con particiones seguras según batch_size.
        """
        try:
            batch_size = self.params.get('write_batch_size', 100000)
            self.logger.info(
                f"Writing data | write_batch_size={batch_size}"
            )

            df.write \
              .mode("overwrite") \
              .option("maxRecordsPerFile", batch_size) \
              .parquet(output_path)

            self.logger.info(f"Data written successfully to {output_path}")

        except Exception as e:
            self.logger.error("Failed writing data", exc_info=True)
            raise


def main(): 

    schema = StructType([
        StructField("src_port", LongType(), True),
        StructField("dst_port", LongType(), True),
        StructField("protocol", StringType(), True),
        StructField("packet_count", LongType(), True),
        StructField("conn_state", StringType(), True),
        StructField("bytes_transferred", DoubleType(), True),
        StructField("timestamp", LongType(), True),
        StructField("attack", LongType(), True)
    ])


    data_dir = "s3a://k8s-mlops-platform-bucket/v1/raw/" #Para Spark
    output_dir = "s3a://k8s-mlops-platform-bucket/v1/processed/" #Para Spark
    artifacts_dir = "s3a://k8s-mlops-platform-bucket/v1/artifacts/"

    preprocessing = SparkPreprocessing(
        schema=schema,
        params_path="/app/repo/k3s/params.yaml",
        data_dir=data_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir
    )

    # 1) Preprocess TRAIN and fit scaler
    preprocessing.preprocess('train')

    # 2) Preprocess VAL using train-fitted transforms
    preprocessing.preprocess('val')

    # 3) Preprocess TEST using train-fitted transforms
    preprocessing.preprocess('test')

if __name__ == "__main__":
    main()
