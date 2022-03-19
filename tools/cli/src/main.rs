use anyhow::Result;
use clap::Parser;
use stdvis_cli::Cli;

fn main() -> Result<()> {
    let cli = Cli::parse();

    cli.command().execute()?;
    Ok(())
}
