from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

from oatomobile.datasets.carla import CARLADataset

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="input_dir",
    default="data/raw",
    help="The full path to the input directory.",
)
flags.DEFINE_string(
    name="output_dir",
    default="data/dataset",
    help="The full path to the output directory.",
)

def main(argv):
  # Debugging purposes.
  logging.debug(argv)
  logging.debug(FLAGS)

  flags.mark_flag_as_required("input_dir")
  flags.mark_flag_as_required("output_dir")

  # Parses command line arguments.
  input_dir = FLAGS.input_dir  
  output_dir = FLAGS.output_dir
  CARLADataset.process(input_dir, output_dir)

if __name__ == "__main__":
  app.run(main)
