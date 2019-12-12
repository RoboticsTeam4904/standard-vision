#[cfg(test)]
extern crate png;

use std::io;

use opencv::core;
use opencv::prelude::*;
use opencv::videoio::*;
use standard_vision::traits::Camera;
use standard_vision::types::{CameraConfig, Image, Pose};

pub struct OpenCVCamera {
    config: CameraConfig,
    video_capture: VideoCapture,
}

impl OpenCVCamera {
    pub fn new_from_index(index: i32, pose: Pose, fov: f64) -> io::Result<Self> {
        let mut video_capture = VideoCapture::default().unwrap();

        if !video_capture.open_with_backend(index, CAP_ANY).unwrap() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Failed to open camera on port {}", index),
            ));
        }

        Self::new_from_video_capture(video_capture, index, pose, fov)
    }

    pub(crate) fn new_from_video_capture(
        video_capture: VideoCapture,
        index: i32,
        pose: Pose,
        fov: f64,
    ) -> io::Result<Self> {
        let resolution = (
            video_capture.get(CAP_PROP_FRAME_WIDTH).unwrap() as u32,
            video_capture.get(CAP_PROP_FRAME_HEIGHT).unwrap() as u32,
        );

        let config = CameraConfig {
            id: index as u8,
            resolution: resolution,
            pose: pose,
            fov: fov,
        };

        Ok(Self {
            config,
            video_capture,
        })
    }

    pub(crate) fn extract_pixels_from_mat(mat: &Mat) -> Vec<[u8; 3]> {
        // Expecting 8UC3 Mat type...
        assert_eq!(mat.typ().unwrap(), core::CV_8UC3);

        let mut pixels = vec![];

        for row_idx in 0..mat.rows().unwrap() {
            for col_idx in 0..mat.cols().unwrap() {
                let pixel_vec3b = *mat.at_2d::<core::Vec3b>(row_idx, col_idx).unwrap();
                pixels.push(*pixel_vec3b);
            }
        }

        pixels
    }
}

impl Camera for OpenCVCamera {
    fn get_config(&self) -> &CameraConfig {
        &self.config
    }

    fn grab_frame(&mut self) -> io::Result<Image> {
        let mut mat = Mat::default().unwrap();
        if !self.video_capture.read(&mut mat).unwrap() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Failed to read frame from camera at port {}",
                    self.config.id
                ),
            ));
        }

        let pixels = Self::extract_pixels_from_mat(&mat);

        Ok(Image {
            timestamp: std::time::SystemTime::now(),
            camera: self.get_config(),
            pixels: pixels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_pixel_extraction() {
        use std::fs::File;
        use opencv::imgcodecs;

        const PATH: &str = "tests/images/rand.png";

        let decoder = png::Decoder::new(File::open(PATH).unwrap());
        let (info, mut reader) = decoder.read_info().unwrap();
        let mut img_buf = vec![0; info.buffer_size()];
        reader.next_frame(&mut img_buf).unwrap();

        let expected_pixels = img_buf
            .chunks_exact(3)
            .map(|items| { [items[2], items[1], items[0]] })
            .collect::<Vec<_>>();

        let mat = imgcodecs::imread(PATH, imgcodecs::IMREAD_COLOR).unwrap();

        let actual_pixels = OpenCVCamera::extract_pixels_from_mat(&mat);
        assert_eq!(expected_pixels, actual_pixels);
    }
}
