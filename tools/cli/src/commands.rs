use anyhow::Result;
use clap::Subcommand;

mod calibrate;
mod sample;

use self::{calibrate::Calibrate, sample::Sample};

#[derive(Subcommand)]
pub enum Commands {
    /// Perform camera calibration given a set of input images with a predefined target
    Calibrate(Calibrate),

    /// Perform configurable bulk-sampling of images
    Sample(Sample),
}

impl Commands {
    pub fn execute(self) -> Result<()> {
        match self {
            Commands::Calibrate(calibrate) => calibrate.execute(),
            Commands::Sample(sample) => sample.execute(),
        }
    }
}
