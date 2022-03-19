use clap::Parser;

mod commands;

use self::commands::Commands;

/// A suite of command-line tools for performing image sampling, calibration, testing, and various other vision-related tasks
#[derive(Parser)]
#[clap(version, about)]
#[clap(propagate_version = true)]
pub struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

impl Cli {
    pub fn command(self) -> Commands {
        self.command
    }
}
