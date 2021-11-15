# Download OATML dataset.
from oatomobile.datasets import CARLADataset

from absl import app
from absl import flags
from absl import logging

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="id",
    default="raw",
    help="id: One of {raw, examples, processed}.",
)
flags.DEFINE_string(
    name="output_dir",
    default="data/oatml",
    help="The path of the datasets.",
)

def main(argv):
  # Debugging purposes.
  logging.debug(argv)
  logging.debug(FLAGS)

  # Parses command line arguments.
  id = FLAGS.id
  output_dir = FLAGS.output_dir

  dataset = CARLADataset(id)
  dataset.download_and_prepare(output_dir)

if __name__ == "__main__":
  flags.mark_flag_as_required("id")
  flags.mark_flag_as_required("output_dir")
  app.run(main)

