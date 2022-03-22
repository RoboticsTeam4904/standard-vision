use std::{fs, io::prelude::*, path::PathBuf, time::Duration};

use anyhow::{bail, Context, Result};
use clap::Parser;
use opencv::imgcodecs;
use serde::{Deserialize, Serialize};
use serde_json;
use stdvis_core::{
    traits::Camera,
    types::{CameraConfig, VisionTarget},
};
use stdvis_opencv::{camera::OcvCamera, convert::AsMatView};

#[derive(Default, Serialize, Deserialize)]
struct Params {
    label: String,
    target: VisionTarget,
    camera: CameraConfig,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    images: Vec<ImageMetadata>,
}

#[derive(Serialize, Deserialize)]
struct ImageMetadata {
    index: usize,
    label: String,
    config: CameraConfig,
    exposure: i32,
}

#[derive(Debug, Parser)]
#[clap(about)]
pub struct Sample {
    /// Specifies the amount of time, in ms, to wait between captures
    #[clap(short, long, name = "delay")]
    delay_ms: Option<u64>,

    /// The path to read input parameters for
    /// If the target file does not exist, a template will be created upon the first run
    #[clap(parse(from_os_str))]
    params_file: PathBuf,

    /// The directory to output images and corresponding metadata to
    #[clap(parse(from_os_str))]
    output_dir: PathBuf,
}

impl Sample {
    // TODO: these should be configurable
    const OUTPUT_FORMAT: &'static str = "png";
    const METADATA_FILENAME: &'static str = "metadata.json";

    fn delay(&self) -> Option<Duration> {
        self.delay_ms.map(|ms| Duration::from_millis(ms))
    }

    pub fn execute(self) -> Result<()> {
        let mut params_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.params_file)
            .context("failed open params file")?;

        let mut params_str = String::new();
        params_file.read_to_string(&mut params_str).unwrap();

        if params_str.is_empty() {
            serde_json::to_writer_pretty(params_file, &Params::default()).unwrap();

            bail!(
                "Please configure input parameters.
                A template file has been created at the specified path."
            );
        }

        let params: Params = match serde_json::from_str(&params_str) {
            Ok(params) => params,
            Err(_) => {
                bail!(
                    "Failed to parse input parameters.
                    The config file may be malformed or out of date.
                    You may want to let the script generate a new template."
                );
            }
        };

        let mut camera = OcvCamera::new(params.camera).unwrap();
        let camera_config = camera.config().clone();

        use std::thread;

        // Allow the camera to "warm up."
        thread::sleep(Duration::from_millis(1000));

        let output_dir = &self.output_dir;

        let mut metadata_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(output_dir.join(Self::METADATA_FILENAME))
            .unwrap();

        let mut metadata_str = String::new();
        metadata_file.read_to_string(&mut metadata_str).unwrap();

        let mut metadata =
            serde_json::from_str(&metadata_str).unwrap_or(Metadata { images: Vec::new() });

        for idx in 0..10 {
            // TODO: scale exposure based on min and max (and add configurability)
            camera.set_exposure(idx * 20).unwrap();

            let frame = camera.grab_frame().unwrap();
            let image_mat = frame.as_mat_view();

            let index = metadata.images.len();

            imgcodecs::imwrite(
                output_dir
                    .join(format!("{}_{}", params.label, index))
                    .with_extension(Self::OUTPUT_FORMAT)
                    .to_str()
                    .unwrap(),
                &*image_mat,
                &opencv::types::VectorOfi32::with_capacity(0),
            )
            .context("writing image to disk")?;

            metadata.images.push(ImageMetadata {
                index,
                label: params.label.clone(),
                config: camera_config.clone(),
                exposure: camera.exposure().unwrap().clone(),
            });

            if let Some(delay) = self.delay() {
                thread::sleep(delay);
            }
        }

        let metadata_file = fs::OpenOptions::new()
            .truncate(true)
            .write(true)
            .open(output_dir.join(Self::METADATA_FILENAME))?;

        serde_json::to_writer_pretty(metadata_file, &metadata)?;

        Ok(())
    }
}
