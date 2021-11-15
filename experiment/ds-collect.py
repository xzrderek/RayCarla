# Download OATML dataset.
from oatomobile.datasets import CARLADataset

from absl import app
from absl import flags
from absl import logging

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="town",
    default="Town01",
    help="town: one of {Town01, Town02, Town03, Town04, Town05}",
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
  town = FLAGS.town
  output_dir = FLAGS.output_dir

  dataset = CARLADataset("raw")
  dataset.collect(town=town, output_dir=output_dir, num_vehicles=10, num_pedestrians=10, render=True)

if __name__ == "__main__":
  flags.mark_flag_as_required("town")
  flags.mark_flag_as_required("output_dir")
  app.run(main)

