use std::{fs, io::prelude::*, path::PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use opencv::{
    calib3d::{
        self, CALIB_CB_ACCURACY, CALIB_CB_EXHAUSTIVE, CALIB_CB_LARGER, CALIB_CB_MARKER,
        CALIB_CB_NORMALIZE_IMAGE,
    },
    core::{Mat, Point2f, Point3f, Size, TermCriteria, TermCriteria_Type, Vector},
    imgcodecs::{self, IMREAD_COLOR},
    prelude::*,
};

use stdvis_core::types::CameraConfig;
use stdvis_opencv::convert::AsArrayView;

const MIN_CALIBRATION_IMAGES: usize = 3;

#[derive(Debug, Parser)]
#[clap(about)]
pub struct Calibrate {
    /// Side length of a checkerboard square in millimeters
    #[clap(short = 's', long = "square-size")]
    square_size_mm: f32,

    /// Number of checkerboard rows to try to detect
    #[clap(short = 'r', long = "board-rows")]
    board_rows: u8,

    /// Number of checkerboard columns to try to detect
    #[clap(short = 'c', long = "board-cols")]
    board_cols: u8,

    /// Path to write resulting camera configuration file
    #[clap(short = 'o', long = "config", parse(from_os_str))]
    camera_config: PathBuf,

    /// If specified, path to directory for writing images with detected checkerboard markers overlaid
    #[clap(short, long, parse(from_os_str))]
    debug_dir: Option<PathBuf>,

    /// Paths to images containing checkerboards to be used for calibration
    #[clap(parse(from_os_str), min_values = MIN_CALIBRATION_IMAGES, required = true)]
    image_paths: Vec<PathBuf>,
}

impl Calibrate {
    fn compute_checkerboard_obj_points(&self) -> Vector<Point3f> {
        let square_size = self.square_size_mm / 1000.0;
        let width = self.width();
        let height = self.height();

        let mut obj_points = Vector::with_capacity((width * height) as usize);

        for col in 0..width {
            for row in 0..height {
                obj_points.push(Point3f::new(
                    square_size * col as f32,
                    square_size * row as f32,
                    0.,
                ));
            }
        }

        obj_points
    }

    fn width(&self) -> u8 {
        self.board_cols - 1
    }

    fn height(&self) -> u8 {
        self.board_rows - 1
    }

    pub fn execute(self) -> Result<()> {
        let template_obj_points = self.compute_checkerboard_obj_points();

        let num_images = self.image_paths.len();
        let mut object_points = Vector::<Vector<Point3f>>::with_capacity(num_images);
        let mut image_points = Vector::<Vector<Point2f>>::with_capacity(num_images);

        let mut image_size = None;
        let pattern_size = Size::new(self.width() as i32, self.height() as i32);

        for path in self.image_paths {
            let image = imgcodecs::imread(path.to_str().unwrap(), IMREAD_COLOR)
                .context("reading image from disk")?;

            let curr_image_size = image.size().context("expected image to have size")?;

            if image_size.get_or_insert(curr_image_size) != &curr_image_size {
                bail!("Expected all images to be the same size. Failed on image: {path:?}");
            }

            let mut corners = Vector::<Point2f>::new();
            let found = calib3d::find_chessboard_corners_sb(
                &image,
                pattern_size,
                &mut corners,
                CALIB_CB_NORMALIZE_IMAGE
                    | CALIB_CB_EXHAUSTIVE
                    | CALIB_CB_ACCURACY
                    | CALIB_CB_MARKER
                    | CALIB_CB_LARGER,
            )
            .context("finding checkerboard corners")?;

            if let Some(ref out_path) = self.debug_dir {
                let mut image_debug = image.clone();

                calib3d::draw_chessboard_corners(&mut image_debug, pattern_size, &corners, found)
                    .context("drawing checkerboard corners")?;

                imgcodecs::imwrite(
                    out_path.join(path.file_name().unwrap()).to_str().unwrap(),
                    &image_debug,
                    &Vector::new(),
                )
                .context("writing debug image to disk")?;
            }

            if !found {
                println!("Failed to find checkerboard corners for image: {path:?}");
                continue;
            }

            let mut sharpness = opencv::core::no_array();
            let sharpness_stats = calib3d::estimate_chessboard_sharpness(
                &image,
                pattern_size,
                &corners,
                0.8,
                false,
                &mut sharpness,
            )
            .context("estimating checkerboard sharpness")?;

            println!(
                "avg. sharpness: {}, avg. brightness (min, max): ({}, {}) for image: {path:?}",
                sharpness_stats[0], sharpness_stats[1], sharpness_stats[2]
            );

            object_points.push(template_obj_points.clone());
            image_points.push(corners);
        }

        println!(
            "Successfully performed corner-finding on {} images",
            image_points.len()
        );

        if image_points.len() < MIN_CALIBRATION_IMAGES {
            bail!("Insufficient successful corner-finding results to continue to calibration");
        }

        let image_size = image_size.expect("image_size should be Some");

        // Use the top-right corner of the checkerboard as a fixed point, as recommended by the documentation for `calibrate_camera_ro`.
        let i_fixed_point = (template_obj_points.len() - 2) as i32;

        let mut camera_matrix = Mat::default();
        let mut dist_coeffs = Mat::default();
        let mut rvecs = Mat::default();
        let mut tvecs = Mat::default();
        let mut new_obj_points = opencv::core::no_array();

        let reproj_error = calib3d::calibrate_camera_ro(
            &object_points,
            &image_points,
            image_size,
            i_fixed_point,
            &mut camera_matrix,
            &mut dist_coeffs,
            &mut rvecs,
            &mut tvecs,
            &mut new_obj_points,
            0,
            // Listed as the default TermCriteria for this method in the OpenCV docs.
            TermCriteria::new(
                TermCriteria_Type::COUNT as i32 + TermCriteria_Type::EPS as i32,
                30,
                std::f64::EPSILON,
            )
            .unwrap(),
        )
        .context("calibrating camera")?;

        println!("Calibration finished with reprojection error: {reproj_error}");

        let mut config_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.camera_config)
            .context("opening config file")?;

        let mut config_str = String::new();
        config_file
            .read_to_string(&mut config_str)
            .context("reading camera config file")?;

        let mut config = if config_str.is_empty() {
            CameraConfig::default()
        } else {
            match serde_json::from_str(&config_str) {
                Ok(config) => config,
                Err(_) => {
                    panic!(
                        "Failed to parse camera configuration; the schema used may be out of date"
                    );
                }
            }
        };

        config.intrinsic_matrix = camera_matrix
            .as_array_view::<f64>()
            .into_shape((3, 3))
            .context("converting intrinsic_matrix Mat")?
            .to_owned();

        config.distortion_coeffs = dist_coeffs
            .as_array_view::<f64>()
            .into_shape(5)
            .context("converting distortion_coeffs Mat")?
            .to_owned();

        serde_json::to_writer_pretty(config_file, &config)
            .context("writing updated config file")?;

        Ok(())
    }
}
