import tensorflow as tf
from tfx import v1 as tfx
import kfp
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import logging
from typing import List, Optional
import os

GOOGLE_CLOUD_PROJECT = 'dt-ml-pipeline'         
GOOGLE_CLOUD_PROJECT_NUMBER = '994103403822'  
GOOGLE_CLOUD_REGION = 'us-central1'          
GCS_BUCKET_NAME = 'dt-ml-pipeline-bucket' 
trainer_module_file = 'penguin_trainer.py'
PIPELINE_NAME = 'ct-bigquery'

# Path to various pipeline artifact.
PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(
    GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' Python module.
MODULE_ROOT = 'gs://{}/pipeline_module/{}'.format(
    GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' data.
DATA_ROOT = 'gs://{}/data/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# This is the path where your model will be pushed for serving.
SERVING_MODEL_DIR = 'gs://{}/serving_model/{}'.format(
    GCS_BUCKET_NAME, PIPELINE_NAME)

QUERY = "SELECT * FROM `tfx-oss-public.palmer_penguins.palmer_penguins`"


def _create_pipeline(pipeline_name: str, pipeline_root: str, query: str,
                     module_file: str, serving_model_dir: str,
                     beam_pipeline_args: Optional[List[str]],
                     ) -> tfx.dsl.Pipeline:
  """Creates a TFX pipeline using BigQuery."""

  # NEW: Query data in BigQuery as a data source.
  example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
      query=query)

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],
      train_args=tfx.proto.TrainArgs(num_steps=100),
      eval_args=tfx.proto.EvalArgs(num_steps=5))

  # Pushes the model to a file destination.
  pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  components = [
      example_gen,
      trainer,
      pusher,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # NEW: `beam_pipeline_args` is required to use BigQueryExampleGen.
      beam_pipeline_args=beam_pipeline_args)
	  

BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
   '--project=' + GOOGLE_CLOUD_PROJECT,
   '--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp'),
   ]
PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'

runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
    output_filename=PIPELINE_DEFINITION_FILE)
_ = runner.run(
    _create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        query=QUERY,
        module_file=os.path.join(MODULE_ROOT, trainer_module_file),
        serving_model_dir=SERVING_MODEL_DIR,
        beam_pipeline_args=BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS))
		

logging.getLogger().setLevel(logging.INFO)

aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)

job = pipeline_jobs.PipelineJob(template_path=PIPELINE_DEFINITION_FILE,
                                display_name=PIPELINE_NAME)
job.submit()
