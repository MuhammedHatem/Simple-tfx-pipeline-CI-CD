import tensorflow as tf
from tfx import v1 as tfx
import kfp
from typing import List, Optional
import absl
import tensorflow_model_analysis as tfma
from tfx.components.trainer.executor import Executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing 
GOOGLE_CLOUD_PROJECT = 'dt-ml-pipeline'    
GOOGLE_CLOUD_PROJECT_NUMBER = '994103403822' 
GOOGLE_CLOUD_REGION = 'us-central1'        
GCS_BUCKET_NAME = 'dt-ml-pipeline-bucket' 
QUERY = "SELECT * FROM `dt-ml-pipeline.crime2021.chicago`"

PIPELINE_NAME = 'my-tfx'

ENDPOINT_NAME = 'prediction-' + PIPELINE_NAME

module_file = 'Chicago_utils.py'
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

print('PIPELINE_ROOT: {}'.format(PIPELINE_ROOT))

def _create_pipeline(pipeline_name: str, pipeline_root: str, query: str,
                     module_file: str, serving_model_dir: str,
                     beam_pipeline_args: Optional[List[str]],
                     ) -> tfx.dsl.Pipeline:
  """Creates a TFX pipeline using BigQuery."""

  # NEW: Query data in BigQuery as a data source.
  example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
      query=query)
  
  statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])
    
  schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'])
  
  example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])
    
  transform = tfx.components.Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)
   
   
  latest_model_resolver = resolver.Resolver(
      strategy_class=latest_artifacts_resolver.LatestArtifactsResolver,
      latest_model=Channel(type=Model)).with_id('latest_model_resolver')
  
  
  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
    module_file=module_file,
    base_model=latest_model_resolver.outputs['latest_model'],
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval']))
  
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='eval')],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['case_number_y'])
      ],
      metrics_specs=
          tfma.metrics.default_regression_specs(
                ))
  
  model_resolver = resolver.Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(
          type=ModelBlessing)).with_id('latest_blessed_model_resolver')
    
  evaluator = tfx.components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)


  # Pushes the model to a file destination.
  pusher = tfx.components.Pusher(
      model_blessing=evaluator.outputs['blessing'],
      model=trainer.outputs['model'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  components = [
      example_gen,
      statistics_gen,
      schema_gen,
      example_validator,
      transform,
      latest_model_resolver,
      trainer,
      model_resolver,
      evaluator,
      pusher,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # NEW: `beam_pipeline_args` is required to use BigQueryExampleGen.
      beam_pipeline_args=beam_pipeline_args)
      
import os

# We need to pass some GCP related configs to BigQuery. This is currently done
# using `beam_pipeline_args` parameter.
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
        module_file=os.path.join(MODULE_ROOT,module_file),
        #os.path.join(MODULE_ROOT,module_file),
        serving_model_dir=SERVING_MODEL_DIR,
        beam_pipeline_args=BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS))
        
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import logging
logging.getLogger().setLevel(logging.INFO)

aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)

job = pipeline_jobs.PipelineJob(template_path=PIPELINE_DEFINITION_FILE,
                                display_name=PIPELINE_NAME)
job.submit()
